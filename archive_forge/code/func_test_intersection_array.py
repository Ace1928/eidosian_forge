import networkx as nx
from networkx import is_strongly_regular
def test_intersection_array(self):
    b, c = nx.intersection_array(nx.cycle_graph(5))
    assert b == [2, 1]
    assert c == [1, 1]
    b, c = nx.intersection_array(nx.dodecahedral_graph())
    assert b == [3, 2, 1, 1, 1]
    assert c == [1, 1, 1, 2, 3]
    b, c = nx.intersection_array(nx.icosahedral_graph())
    assert b == [5, 2, 1]
    assert c == [1, 2, 5]