import networkx as nx
from networkx.utils import not_implemented_for
def restore_node(self):
    self.g.add_node(self.original)
    for outer in self.outer_vertices:
        adj_view = self.g[outer]
        for neighbor, edge_attrs in adj_view.items():
            if neighbor not in self.core_vertices:
                self.g.add_edge(self.original, neighbor, **edge_attrs)
                break
    self.g.remove_nodes_from(self.outer_vertices)
    self.g.remove_nodes_from(self.inner_vertices)
    self.g.remove_nodes_from(self.core_vertices)