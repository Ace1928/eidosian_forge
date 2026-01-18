from collections import defaultdict, deque
from itertools import chain, combinations, islice
import networkx as nx
from networkx.utils import not_implemented_for
def find_branching_nodes(self, P, target):
    """Find a set of nodes to branch on."""
    residual_wt = {v: self.node_weights[v] for v in P}
    total_wt = 0
    P = P[:]
    while P:
        independent_set = self.greedily_find_independent_set(P)
        min_wt_in_class = min((residual_wt[v] for v in independent_set))
        total_wt += min_wt_in_class
        if total_wt > target:
            break
        for v in independent_set:
            residual_wt[v] -= min_wt_in_class
        P = [v for v in P if residual_wt[v] != 0]
    return P