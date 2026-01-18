import sys
from django.apps import apps
from django.core.management.base import BaseCommand
from django.db import DEFAULT_DB_ALIAS, connections
from django.db.migrations.loader import MigrationLoader
from django.db.migrations.recorder import MigrationRecorder
def show_plan(self, connection, app_names=None):
    """
        Show all known migrations (or only those of the specified app_names)
        in the order they will be applied.
        """
    loader = MigrationLoader(connection)
    graph = loader.graph
    if app_names:
        self._validate_app_names(loader, app_names)
        targets = [key for key in graph.leaf_nodes() if key[0] in app_names]
    else:
        targets = graph.leaf_nodes()
    plan = []
    seen = set()
    for target in targets:
        for migration in graph.forwards_plan(target):
            if migration not in seen:
                node = graph.node_map[migration]
                plan.append(node)
                seen.add(migration)

    def print_deps(node):
        out = []
        for parent in sorted(node.parents):
            out.append('%s.%s' % parent.key)
        if out:
            return ' ... (%s)' % ', '.join(out)
        return ''
    for node in plan:
        deps = ''
        if self.verbosity >= 2:
            deps = print_deps(node)
        if node.key in loader.applied_migrations:
            self.stdout.write('[X]  %s.%s%s' % (node.key[0], node.key[1], deps))
        else:
            self.stdout.write('[ ]  %s.%s%s' % (node.key[0], node.key[1], deps))
    if not plan:
        self.stdout.write('(no migrations)', self.style.ERROR)