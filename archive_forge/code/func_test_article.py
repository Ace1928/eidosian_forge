import networkx as nx
def test_article(self):
    """our algorithm matches article's"""
    G = small_ego_G()
    disp_uh = nx.dispersion(G, 'u', 'h', normalized=False)
    disp_ub = nx.dispersion(G, 'u', 'b', normalized=False)
    assert disp_uh == 4
    assert disp_ub == 1