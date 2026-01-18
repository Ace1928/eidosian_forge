from itertools import permutations
import pytest
import networkx as nx
def test_in_out_weight(self):
    G = nx.DiGraph()
    G.add_edge(1, 2, weight=1)
    G.add_edge(1, 3, weight=1)
    G.add_edge(3, 1, weight=1)
    for s, t in permutations(['in', 'out', 'in+out'], 2):
        c = nx.average_degree_connectivity(G, source=s, target=t)
        cw = nx.average_degree_connectivity(G, source=s, target=t, weight='weight')
        assert c == cw