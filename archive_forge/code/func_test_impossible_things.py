import networkx as nx
def test_impossible_things(self):
    G = nx.karate_club_graph()
    disp = nx.dispersion(G)
    for u in disp:
        for v in disp[u]:
            assert disp[u][v] >= 0