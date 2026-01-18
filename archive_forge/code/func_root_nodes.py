from functools import total_ordering
from django.db.migrations.state import ProjectState
from .exceptions import CircularDependencyError, NodeNotFoundError
def root_nodes(self, app=None):
    """
        Return all root nodes - that is, nodes with no dependencies inside
        their app. These are the starting point for an app.
        """
    roots = set()
    for node in self.nodes:
        if all((key[0] != node[0] for key in self.node_map[node].parents)) and (not app or app == node[0]):
            roots.add(node)
    return sorted(roots)