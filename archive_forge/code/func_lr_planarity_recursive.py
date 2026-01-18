from collections import defaultdict
import networkx as nx
def lr_planarity_recursive(self):
    """Recursive version of :meth:`lr_planarity`."""
    if self.G.order() > 2 and self.G.size() > 3 * self.G.order() - 6:
        return None
    for v in self.G:
        if self.height[v] is None:
            self.height[v] = 0
            self.roots.append(v)
            self.dfs_orientation_recursive(v)
    self.G = None
    for v in self.DG:
        self.ordered_adjs[v] = sorted(self.DG[v], key=lambda x: self.nesting_depth[v, x])
    for v in self.roots:
        if not self.dfs_testing_recursive(v):
            return None
    for e in self.DG.edges:
        self.nesting_depth[e] = self.sign_recursive(e) * self.nesting_depth[e]
    self.embedding.add_nodes_from(self.DG.nodes)
    for v in self.DG:
        self.ordered_adjs[v] = sorted(self.DG[v], key=lambda x: self.nesting_depth[v, x])
        previous_node = None
        for w in self.ordered_adjs[v]:
            self.embedding.add_half_edge_cw(v, w, previous_node)
            previous_node = w
    for v in self.roots:
        self.dfs_embedding_recursive(v)
    return self.embedding