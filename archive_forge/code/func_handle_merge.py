import os
import sys
import warnings
from itertools import takewhile
from django.apps import apps
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError, no_translations
from django.core.management.utils import run_formatters
from django.db import DEFAULT_DB_ALIAS, OperationalError, connections, router
from django.db.migrations import Migration
from django.db.migrations.autodetector import MigrationAutodetector
from django.db.migrations.loader import MigrationLoader
from django.db.migrations.migration import SwappableTuple
from django.db.migrations.optimizer import MigrationOptimizer
from django.db.migrations.questioner import (
from django.db.migrations.state import ProjectState
from django.db.migrations.utils import get_migration_name_timestamp
from django.db.migrations.writer import MigrationWriter
def handle_merge(self, loader, conflicts):
    """
        Handles merging together conflicted migrations interactively,
        if it's safe; otherwise, advises on how to fix it.
        """
    if self.interactive:
        questioner = InteractiveMigrationQuestioner(prompt_output=self.log_output)
    else:
        questioner = MigrationQuestioner(defaults={'ask_merge': True})
    for app_label, migration_names in conflicts.items():
        merge_migrations = []
        for migration_name in migration_names:
            migration = loader.get_migration(app_label, migration_name)
            migration.ancestry = [mig for mig in loader.graph.forwards_plan((app_label, migration_name)) if mig[0] == migration.app_label]
            merge_migrations.append(migration)

        def all_items_equal(seq):
            return all((item == seq[0] for item in seq[1:]))
        merge_migrations_generations = zip(*(m.ancestry for m in merge_migrations))
        common_ancestor_count = sum((1 for common_ancestor_generation in takewhile(all_items_equal, merge_migrations_generations)))
        if not common_ancestor_count:
            raise ValueError('Could not find common ancestor of %s' % migration_names)
        for migration in merge_migrations:
            migration.branch = migration.ancestry[common_ancestor_count:]
            migrations_ops = (loader.get_migration(node_app, node_name).operations for node_app, node_name in migration.branch)
            migration.merged_operations = sum(migrations_ops, [])
        if self.verbosity > 0:
            self.log(self.style.MIGRATE_HEADING('Merging %s' % app_label))
            for migration in merge_migrations:
                self.log(self.style.MIGRATE_LABEL('  Branch %s' % migration.name))
                for operation in migration.merged_operations:
                    self.log('    - %s' % operation.describe())
        if questioner.ask_merge(app_label):
            numbers = [MigrationAutodetector.parse_number(migration.name) for migration in merge_migrations]
            try:
                biggest_number = max((x for x in numbers if x is not None))
            except ValueError:
                biggest_number = 1
            subclass = type('Migration', (Migration,), {'dependencies': [(app_label, migration.name) for migration in merge_migrations]})
            parts = ['%04i' % (biggest_number + 1)]
            if self.migration_name:
                parts.append(self.migration_name)
            else:
                parts.append('merge')
                leaf_names = '_'.join(sorted((migration.name for migration in merge_migrations)))
                if len(leaf_names) > 47:
                    parts.append(get_migration_name_timestamp())
                else:
                    parts.append(leaf_names)
            migration_name = '_'.join(parts)
            new_migration = subclass(migration_name, app_label)
            writer = MigrationWriter(new_migration, self.include_header)
            if not self.dry_run:
                with open(writer.path, 'w', encoding='utf-8') as fh:
                    fh.write(writer.as_string())
                run_formatters([writer.path])
                if self.verbosity > 0:
                    self.log('\nCreated new merge migration %s' % writer.path)
                    if self.scriptable:
                        self.stdout.write(writer.path)
            elif self.verbosity == 3:
                self.log(self.style.MIGRATE_HEADING("Full merge migrations file '%s':" % writer.filename))
                self.log(writer.as_string())