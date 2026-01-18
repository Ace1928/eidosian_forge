import itertools
import pytest
import networkx as nx
def rs_shc():
    graph = nx.Graph()
    graph.add_nodes_from([1, 2, 3, 4])
    graph.add_edges_from([(1, 2), (2, 3), (3, 4)])
    return graph