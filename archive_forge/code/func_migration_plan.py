from django.apps.registry import apps as global_apps
from django.db import migrations, router
from .exceptions import InvalidMigrationPlan
from .loader import MigrationLoader
from .recorder import MigrationRecorder
from .state import ProjectState
def migration_plan(self, targets, clean_start=False):
    """
        Given a set of targets, return a list of (Migration instance, backwards?).
        """
    plan = []
    if clean_start:
        applied = {}
    else:
        applied = dict(self.loader.applied_migrations)
    for target in targets:
        if target[1] is None:
            for root in self.loader.graph.root_nodes():
                if root[0] == target[0]:
                    for migration in self.loader.graph.backwards_plan(root):
                        if migration in applied:
                            plan.append((self.loader.graph.nodes[migration], True))
                            applied.pop(migration)
        elif target in applied:
            if self.loader.replace_migrations and target not in self.loader.graph.node_map:
                self.loader.replace_migrations = False
                self.loader.build_graph()
                return self.migration_plan(targets, clean_start=clean_start)
            next_in_app = sorted((n for n in self.loader.graph.node_map[target].children if n[0] == target[0]))
            for node in next_in_app:
                for migration in self.loader.graph.backwards_plan(node):
                    if migration in applied:
                        plan.append((self.loader.graph.nodes[migration], True))
                        applied.pop(migration)
        else:
            for migration in self.loader.graph.forwards_plan(target):
                if migration not in applied:
                    plan.append((self.loader.graph.nodes[migration], False))
                    applied[migration] = self.loader.graph.nodes[migration]
    return plan