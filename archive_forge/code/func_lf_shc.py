import itertools
import pytest
import networkx as nx
def lf_shc():
    graph = nx.Graph()
    graph.add_nodes_from([1, 2, 3, 4, 5, 6])
    graph.add_edges_from([(6, 1), (1, 4), (4, 3), (3, 2), (2, 5)])
    return graph