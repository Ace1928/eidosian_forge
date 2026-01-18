from functools import total_ordering
from django.db.migrations.state import ProjectState
from .exceptions import CircularDependencyError, NodeNotFoundError
def forwards_plan(self, target):
    """
        Given a node, return a list of which previous nodes (dependencies) must
        be applied, ending with the node itself. This is the list you would
        follow if applying the migrations to a database.
        """
    if target not in self.nodes:
        raise NodeNotFoundError('Node %r not a valid node' % (target,), target)
    return self.iterative_dfs(self.node_map[target])