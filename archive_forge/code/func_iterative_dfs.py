from functools import total_ordering
from django.db.migrations.state import ProjectState
from .exceptions import CircularDependencyError, NodeNotFoundError
def iterative_dfs(self, start, forwards=True):
    """Iterative depth-first search for finding dependencies."""
    visited = []
    visited_set = set()
    stack = [(start, False)]
    while stack:
        node, processed = stack.pop()
        if node in visited_set:
            pass
        elif processed:
            visited_set.add(node)
            visited.append(node.key)
        else:
            stack.append((node, True))
            stack += [(n, False) for n in sorted(node.parents if forwards else node.children)]
    return visited