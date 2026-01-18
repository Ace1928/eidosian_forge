import itertools
import pytest
import networkx as nx
def sli_hc():
    graph = nx.Graph()
    graph.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9])
    graph.add_edges_from([(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 7), (2, 8), (2, 9), (3, 6), (3, 7), (3, 9), (4, 5), (4, 6), (4, 8), (4, 9), (5, 6), (5, 7), (5, 8), (6, 7), (6, 9), (7, 8), (8, 9)])
    return graph