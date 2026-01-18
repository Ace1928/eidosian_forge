import itertools
import pytest
import networkx as nx
def sl_shc():
    graph = nx.Graph()
    graph.add_nodes_from([1, 2, 3, 4, 5, 6])
    graph.add_edges_from([(1, 2), (1, 3), (2, 3), (1, 4), (2, 5), (3, 6), (4, 5), (4, 6), (5, 6)])
    return graph