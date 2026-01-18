from functools import total_ordering
from django.db.migrations.state import ProjectState
from .exceptions import CircularDependencyError, NodeNotFoundError
def backwards_plan(self, target):
    """
        Given a node, return a list of which dependent nodes (dependencies)
        must be unapplied, ending with the node itself. This is the list you
        would follow if removing the migrations from a database.
        """
    if target not in self.nodes:
        raise NodeNotFoundError('Node %r not a valid node' % (target,), target)
    return self.iterative_dfs(self.node_map[target], forwards=False)