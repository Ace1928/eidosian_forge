import pytest
import networkx as nx
from networkx.generators.classic import barbell_graph, cycle_graph, path_graph
from networkx.utils import graphs_equal
def test_from_numpy_array_multiedge_no_edge_attr(self):
    A = np.array([[0, 2], [2, 0]])
    G = nx.from_numpy_array(A, create_using=nx.MultiDiGraph, edge_attr=None)
    assert all(('weight' not in e for _, e in G[0][1].items()))
    assert len(G[0][1][0]) == 0