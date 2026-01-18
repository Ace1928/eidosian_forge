import itertools
import pytest
import networkx as nx
def slf_shc():
    graph = nx.Graph()
    graph.add_nodes_from([1, 2, 3, 4, 5, 6, 7])
    graph.add_edges_from([(1, 2), (1, 5), (1, 6), (2, 3), (2, 7), (3, 4), (3, 7), (4, 5), (4, 6), (5, 6)])
    return graph