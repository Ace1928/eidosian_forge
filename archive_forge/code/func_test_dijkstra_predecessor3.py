import pytest
import networkx as nx
from networkx.utils import pairwise
def test_dijkstra_predecessor3(self):
    XG = nx.DiGraph()
    XG.add_weighted_edges_from([('s', 'u', 10), ('s', 'x', 5), ('u', 'v', 1), ('u', 'x', 2), ('v', 'y', 1), ('x', 'u', 3), ('x', 'v', 5), ('x', 'y', 2), ('y', 's', 7), ('y', 'v', 6)])
    P, D = nx.dijkstra_predecessor_and_distance(XG, 's')
    assert P['v'] == ['u']
    assert D['v'] == 9
    P, D = nx.dijkstra_predecessor_and_distance(XG, 's', cutoff=8)
    assert 'v' not in D