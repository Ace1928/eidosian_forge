import pytest
import networkx as nx
def test_exception_multiple_graphs(to_latex=nx.to_latex):
    G = nx.path_graph(3)
    pos_bad = {0: (1, 2), 1: (0, 1)}
    pos_OK = {0: (1, 2), 1: (0, 1), 2: (2, 1)}
    fourG = [G, G, G, G]
    fourpos = [pos_OK, pos_OK, pos_OK, pos_OK]
    to_latex(fourG, pos_OK)
    with pytest.raises(nx.NetworkXError):
        to_latex(fourG, pos_bad)
    to_latex(fourG, fourpos)
    with pytest.raises(nx.NetworkXError):
        to_latex(fourG, [pos_bad, pos_bad, pos_bad, pos_bad])
    with pytest.raises(nx.NetworkXError):
        to_latex(fourG, [pos_OK, pos_OK, pos_bad, pos_OK])
    with pytest.raises(nx.NetworkXError):
        to_latex(fourG, fourpos, sub_captions=['hi', 'hi'])
    with pytest.raises(nx.NetworkXError):
        to_latex(fourG, fourpos, sub_labels=['hi', 'hi'])
    to_latex(fourG, fourpos, sub_captions=['hi'] * 4, sub_labels=['lbl'] * 4)