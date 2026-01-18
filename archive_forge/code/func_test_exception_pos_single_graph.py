import pytest
import networkx as nx
def test_exception_pos_single_graph(to_latex=nx.to_latex):
    G = nx.path_graph(4)
    to_latex(G, pos='pos')
    pos = {0: (1, 2), 1: (0, 1), 2: (2, 1)}
    with pytest.raises(nx.NetworkXError):
        to_latex(G, pos)
    pos[3] = (1, 2, 3)
    with pytest.raises(nx.NetworkXError):
        to_latex(G, pos)
    pos[3] = 2
    with pytest.raises(nx.NetworkXError):
        to_latex(G, pos)
    pos[3] = (3, 2)
    to_latex(G, pos)