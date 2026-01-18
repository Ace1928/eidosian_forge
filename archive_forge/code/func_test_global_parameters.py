import networkx as nx
from networkx import is_strongly_regular
def test_global_parameters(self):
    b, c = nx.intersection_array(nx.cycle_graph(5))
    g = nx.global_parameters(b, c)
    assert list(g) == [(0, 0, 2), (1, 0, 1), (1, 1, 0)]
    b, c = nx.intersection_array(nx.cycle_graph(3))
    g = nx.global_parameters(b, c)
    assert list(g) == [(0, 0, 2), (1, 1, 0)]