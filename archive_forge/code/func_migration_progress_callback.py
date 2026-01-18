import sys
import time
from importlib import import_module
from django.apps import apps
from django.core.management.base import BaseCommand, CommandError, no_translations
from django.core.management.sql import emit_post_migrate_signal, emit_pre_migrate_signal
from django.db import DEFAULT_DB_ALIAS, connections, router
from django.db.migrations.autodetector import MigrationAutodetector
from django.db.migrations.executor import MigrationExecutor
from django.db.migrations.loader import AmbiguityError
from django.db.migrations.state import ModelState, ProjectState
from django.utils.module_loading import module_has_submodule
from django.utils.text import Truncator
def migration_progress_callback(self, action, migration=None, fake=False):
    if self.verbosity >= 1:
        compute_time = self.verbosity > 1
        if action == 'apply_start':
            if compute_time:
                self.start = time.monotonic()
            self.stdout.write('  Applying %s...' % migration, ending='')
            self.stdout.flush()
        elif action == 'apply_success':
            elapsed = ' (%.3fs)' % (time.monotonic() - self.start) if compute_time else ''
            if fake:
                self.stdout.write(self.style.SUCCESS(' FAKED' + elapsed))
            else:
                self.stdout.write(self.style.SUCCESS(' OK' + elapsed))
        elif action == 'unapply_start':
            if compute_time:
                self.start = time.monotonic()
            self.stdout.write('  Unapplying %s...' % migration, ending='')
            self.stdout.flush()
        elif action == 'unapply_success':
            elapsed = ' (%.3fs)' % (time.monotonic() - self.start) if compute_time else ''
            if fake:
                self.stdout.write(self.style.SUCCESS(' FAKED' + elapsed))
            else:
                self.stdout.write(self.style.SUCCESS(' OK' + elapsed))
        elif action == 'render_start':
            if compute_time:
                self.start = time.monotonic()
            self.stdout.write('  Rendering model states...', ending='')
            self.stdout.flush()
        elif action == 'render_success':
            elapsed = ' (%.3fs)' % (time.monotonic() - self.start) if compute_time else ''
            self.stdout.write(self.style.SUCCESS(' DONE' + elapsed))