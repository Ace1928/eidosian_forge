from collections import deque
from operator import itemgetter
import networkx as nx
from networkx.algorithms.flow.utils import build_residual_network
def grow():
    """Bidirectional breadth-first search for the growth stage.

        Returns a connecting edge, that is and edge that connects
        a node from the source search tree with a node from the
        target search tree.
        The first node in the connecting edge is always from the
        source tree and the last node from the target tree.
        """
    while active:
        u = active[0]
        if u in source_tree:
            this_tree = source_tree
            other_tree = target_tree
            neighbors = R_succ
        else:
            this_tree = target_tree
            other_tree = source_tree
            neighbors = R_pred
        for v, attr in neighbors[u].items():
            if attr['capacity'] - attr['flow'] > 0:
                if v not in this_tree:
                    if v in other_tree:
                        return (u, v) if this_tree is source_tree else (v, u)
                    this_tree[v] = u
                    dist[v] = dist[u] + 1
                    timestamp[v] = timestamp[u]
                    active.append(v)
                elif v in this_tree and _is_closer(u, v):
                    this_tree[v] = u
                    dist[v] = dist[u] + 1
                    timestamp[v] = timestamp[u]
        _ = active.popleft()
    return (None, None)