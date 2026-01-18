import itertools
import pytest
import networkx as nx
def lf_hc():
    graph = nx.Graph()
    graph.add_nodes_from([1, 2, 3, 4, 5, 6, 7])
    graph.add_edges_from([(1, 7), (1, 6), (1, 3), (1, 4), (7, 2), (2, 6), (2, 3), (2, 5), (5, 3), (5, 4), (4, 3)])
    return graph