import math
import networkx as nx
def test_empty_graph_broadcast():
    H = nx.empty_graph(1)
    b_T, b_C = nx.tree_broadcast_center(H)
    assert b_T == 0
    assert b_C == {0}
    assert nx.tree_broadcast_time(H) == 0