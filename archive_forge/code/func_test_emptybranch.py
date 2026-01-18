import pytest
import networkx as nx
def test_emptybranch():
    G = nx.DiGraph()
    G.add_nodes_from(range(10))
    assert nx.is_branching(G)
    assert not nx.is_arborescence(G)