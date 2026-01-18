import math
import networkx as nx
def test_binomial_tree_broadcast():
    for i in range(2, 8):
        G = nx.binomial_tree(i)
        b_T, b_C = nx.tree_broadcast_center(G)
        assert b_T == i
        assert b_C == {0, 2 ** (i - 1)}
        assert nx.tree_broadcast_time(G) == 2 * i - 1