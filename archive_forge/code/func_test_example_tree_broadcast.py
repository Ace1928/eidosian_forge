import math
import networkx as nx
def test_example_tree_broadcast():
    """
    Test the BROADCAST algorithm on the example in the paper titled: "Information Dissemination in Trees"
    """
    edge_list = [(0, 1), (1, 2), (2, 7), (3, 4), (5, 4), (4, 7), (6, 7), (7, 9), (8, 9), (9, 13), (13, 14), (14, 15), (14, 16), (14, 17), (13, 11), (11, 10), (11, 12), (13, 18), (18, 19), (18, 20)]
    G = nx.Graph(edge_list)
    b_T, b_C = nx.tree_broadcast_center(G)
    assert b_T == 6
    assert b_C == {13, 9}
    assert nx.tree_broadcast_time(G, 17) == 8
    assert nx.tree_broadcast_time(G, 3) == 9
    assert nx.tree_broadcast_time(G) == 10