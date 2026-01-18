import pytest
import networkx as nx
from networkx.generators.classic import barbell_graph, cycle_graph, path_graph
from networkx.utils import graphs_equal
def create_weighted(self, G):
    g = cycle_graph(4)
    G.add_nodes_from(g)
    G.add_weighted_edges_from(((u, v, 10 + u) for u, v in g.edges()))
    return G