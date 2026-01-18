import pytest
import networkx as nx
def test_exception_multigraph():
    G = nx.path_graph(4, create_using=nx.MultiGraph)
    G.add_edge(1, 2)
    with pytest.raises(nx.NetworkXNotImplemented):
        nx.to_latex(G)