from collections import defaultdict
import networkx as nx
def dfs_embedding(self, v):
    """Completes the embedding."""
    dfs_stack = [v]
    ind = defaultdict(lambda: 0)
    while dfs_stack:
        v = dfs_stack.pop()
        for w in self.ordered_adjs[v][ind[v]:]:
            ind[v] += 1
            ei = (v, w)
            if ei == self.parent_edge[w]:
                self.embedding.add_half_edge_first(w, v)
                self.left_ref[v] = w
                self.right_ref[v] = w
                dfs_stack.append(v)
                dfs_stack.append(w)
                break
            elif self.side[ei] == 1:
                self.embedding.add_half_edge_cw(w, v, self.right_ref[w])
            else:
                self.embedding.add_half_edge_ccw(w, v, self.left_ref[w])
                self.left_ref[w] = v