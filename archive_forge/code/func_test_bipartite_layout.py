import pytest
import networkx as nx
def test_bipartite_layout(self):
    G = nx.complete_bipartite_graph(3, 5)
    top, bottom = nx.bipartite.sets(G)
    vpos = nx.bipartite_layout(G, top)
    assert len(vpos) == len(G)
    top_x = vpos[list(top)[0]][0]
    bottom_x = vpos[list(bottom)[0]][0]
    for node in top:
        assert vpos[node][0] == top_x
    for node in bottom:
        assert vpos[node][0] == bottom_x
    vpos = nx.bipartite_layout(G, top, align='horizontal', center=(2, 2), scale=2, aspect_ratio=1)
    assert len(vpos) == len(G)
    top_y = vpos[list(top)[0]][1]
    bottom_y = vpos[list(bottom)[0]][1]
    for node in top:
        assert vpos[node][1] == top_y
    for node in bottom:
        assert vpos[node][1] == bottom_y
    pytest.raises(ValueError, nx.bipartite_layout, G, top, align='foo')