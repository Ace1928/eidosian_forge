from collections import defaultdict
import networkx as nx
def dfs_embedding_recursive(self, v):
    """Recursive version of :meth:`dfs_embedding`."""
    for w in self.ordered_adjs[v]:
        ei = (v, w)
        if ei == self.parent_edge[w]:
            self.embedding.add_half_edge_first(w, v)
            self.left_ref[v] = w
            self.right_ref[v] = w
            self.dfs_embedding_recursive(w)
        elif self.side[ei] == 1:
            self.embedding.add_half_edge_cw(w, v, self.right_ref[w])
        else:
            self.embedding.add_half_edge_ccw(w, v, self.left_ref[w])
            self.left_ref[w] = v