import functools
import re
from collections import defaultdict
from graphlib import TopologicalSorter
from itertools import chain
from django.conf import settings
from django.db import models
from django.db.migrations import operations
from django.db.migrations.migration import Migration
from django.db.migrations.operations.models import AlterModelOptions
from django.db.migrations.optimizer import MigrationOptimizer
from django.db.migrations.questioner import MigrationQuestioner
from django.db.migrations.utils import (
def arrange_for_graph(self, changes, graph, migration_name=None):
    """
        Take a result from changes() and a MigrationGraph, and fix the names
        and dependencies of the changes so they extend the graph from the leaf
        nodes for each app.
        """
    leaves = graph.leaf_nodes()
    name_map = {}
    for app_label, migrations in list(changes.items()):
        if not migrations:
            continue
        app_leaf = None
        for leaf in leaves:
            if leaf[0] == app_label:
                app_leaf = leaf
                break
        if app_leaf is None and (not self.questioner.ask_initial(app_label)):
            for migration in migrations:
                name_map[app_label, migration.name] = (app_label, '__first__')
            del changes[app_label]
            continue
        if app_leaf is None:
            next_number = 1
        else:
            next_number = (self.parse_number(app_leaf[1]) or 0) + 1
        for i, migration in enumerate(migrations):
            if i == 0 and app_leaf:
                migration.dependencies.append(app_leaf)
            new_name_parts = ['%04i' % next_number]
            if migration_name:
                new_name_parts.append(migration_name)
            elif i == 0 and (not app_leaf):
                new_name_parts.append('initial')
            else:
                new_name_parts.append(migration.suggest_name()[:100])
            new_name = '_'.join(new_name_parts)
            name_map[app_label, migration.name] = (app_label, new_name)
            next_number += 1
            migration.name = new_name
    for migrations in changes.values():
        for migration in migrations:
            migration.dependencies = [name_map.get(d, d) for d in migration.dependencies]
    return changes