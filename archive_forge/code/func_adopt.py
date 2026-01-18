from collections import deque
from operator import itemgetter
import networkx as nx
from networkx.algorithms.flow.utils import build_residual_network
def adopt():
    """Adoption stage.

        Reconstruct search trees by adopting or discarding orphans.
        During augmentation stage some edges got saturated and thus
        the source and target search trees broke down to forests, with
        orphans as roots of some of its trees. We have to reconstruct
        the search trees rooted to source and target before we can grow
        them again.
        """
    while orphans:
        u = orphans.popleft()
        if u in source_tree:
            tree = source_tree
            neighbors = R_pred
        else:
            tree = target_tree
            neighbors = R_succ
        nbrs = ((n, attr, dist[n]) for n, attr in neighbors[u].items() if n in tree)
        for v, attr, d in sorted(nbrs, key=itemgetter(2)):
            if attr['capacity'] - attr['flow'] > 0:
                if _has_valid_root(v, tree):
                    tree[u] = v
                    dist[u] = dist[v] + 1
                    timestamp[u] = time
                    break
        else:
            nbrs = ((n, attr, dist[n]) for n, attr in neighbors[u].items() if n in tree)
            for v, attr, d in sorted(nbrs, key=itemgetter(2)):
                if attr['capacity'] - attr['flow'] > 0:
                    if v not in active:
                        active.append(v)
                if tree[v] == u:
                    tree[v] = None
                    orphans.appendleft(v)
            if u in active:
                active.remove(u)
            del tree[u]