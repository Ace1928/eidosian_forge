import collections
import io
import os
import networkx as nx
from networkx.drawing import nx_pydot
def bfs_predecessors_iter(self, n):
    """Iterates breadth first over *all* predecessors of a given node.

        This will go through the nodes predecessors, then the predecessor nodes
        predecessors and so on until no more predecessors are found.

        NOTE(harlowja): predecessor cycles (if they exist) will not be iterated
        over more than once (this prevents infinite iteration).
        """
    visited = set([n])
    queue = collections.deque(self.predecessors(n))
    while queue:
        pred = queue.popleft()
        if pred not in visited:
            yield pred
            visited.add(pred)
            for pred_pred in self.predecessors(pred):
                if pred_pred not in visited:
                    queue.append(pred_pred)