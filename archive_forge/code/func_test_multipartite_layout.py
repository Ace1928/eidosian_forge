import pytest
import networkx as nx
def test_multipartite_layout(self):
    sizes = (0, 5, 7, 2, 8)
    G = nx.complete_multipartite_graph(*sizes)
    vpos = nx.multipartite_layout(G)
    assert len(vpos) == len(G)
    start = 0
    for n in sizes:
        end = start + n
        assert all((vpos[start][0] == vpos[i][0] for i in range(start + 1, end)))
        start += n
    vpos = nx.multipartite_layout(G, align='horizontal', scale=2, center=(2, 2))
    assert len(vpos) == len(G)
    start = 0
    for n in sizes:
        end = start + n
        assert all((vpos[start][1] == vpos[i][1] for i in range(start + 1, end)))
        start += n
    pytest.raises(ValueError, nx.multipartite_layout, G, align='foo')