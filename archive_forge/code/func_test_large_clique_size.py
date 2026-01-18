import networkx as nx
from networkx.algorithms.approximation import (
def test_large_clique_size():
    G = nx.complete_graph(9)
    nx.add_cycle(G, [9, 10, 11])
    G.add_edge(8, 9)
    G.add_edge(1, 12)
    G.add_node(13)
    assert large_clique_size(G) == 9
    G.remove_node(5)
    assert large_clique_size(G) == 8
    G.remove_edge(2, 3)
    assert large_clique_size(G) == 7