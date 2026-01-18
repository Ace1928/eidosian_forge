from collections import defaultdict
import networkx as nx
def dfs_testing(self, v):
    """Test for LR partition."""
    dfs_stack = [v]
    ind = defaultdict(lambda: 0)
    skip_init = defaultdict(lambda: False)
    while dfs_stack:
        v = dfs_stack.pop()
        e = self.parent_edge[v]
        skip_final = False
        for w in self.ordered_adjs[v][ind[v]:]:
            ei = (v, w)
            if not skip_init[ei]:
                self.stack_bottom[ei] = top_of_stack(self.S)
                if ei == self.parent_edge[w]:
                    dfs_stack.append(v)
                    dfs_stack.append(w)
                    skip_init[ei] = True
                    skip_final = True
                    break
                else:
                    self.lowpt_edge[ei] = ei
                    self.S.append(ConflictPair(right=Interval(ei, ei)))
            if self.lowpt[ei] < self.height[v]:
                if w == self.ordered_adjs[v][0]:
                    self.lowpt_edge[e] = self.lowpt_edge[ei]
                elif not self.add_constraints(ei, e):
                    return False
            ind[v] += 1
        if not skip_final:
            if e is not None:
                self.remove_back_edges(e)
    return True