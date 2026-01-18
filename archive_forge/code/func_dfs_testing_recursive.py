from collections import defaultdict
import networkx as nx
def dfs_testing_recursive(self, v):
    """Recursive version of :meth:`dfs_testing`."""
    e = self.parent_edge[v]
    for w in self.ordered_adjs[v]:
        ei = (v, w)
        self.stack_bottom[ei] = top_of_stack(self.S)
        if ei == self.parent_edge[w]:
            if not self.dfs_testing_recursive(w):
                return False
        else:
            self.lowpt_edge[ei] = ei
            self.S.append(ConflictPair(right=Interval(ei, ei)))
        if self.lowpt[ei] < self.height[v]:
            if w == self.ordered_adjs[v][0]:
                self.lowpt_edge[e] = self.lowpt_edge[ei]
            elif not self.add_constraints(ei, e):
                return False
    if e is not None:
        self.remove_back_edges(e)
    return True