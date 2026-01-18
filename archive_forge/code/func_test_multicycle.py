import pytest
import networkx as nx
def test_multicycle():
    G = nx.MultiDiGraph()
    G.add_edges_from([(0, 1), (0, 1)])
    assert not nx.is_tree(G)
    assert nx.is_directed_acyclic_graph(G)