from datetime import datetime, timedelta
import pytest
import networkx as nx
def test_common_graph_with_int_attributes(self):
    G = nx.DiGraph()
    G.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    G.add_edge(4, 2)
    G.add_edge(4, 0)
    G.add_edge(4, 1)
    G.add_edge(4, 3)
    G.add_edge(5, 2)
    G.add_edge(6, 2)
    G.add_edge(6, 4)
    G.add_edge(7, 4)
    G.add_edge(8, 4)
    G.add_edge(9, 4)
    G.add_edge(9, 1)
    G.add_edge(9, 3)
    G.add_edge(10, 4)
    node_attrs = {0: {'time': 20}, 1: {'time': 20}, 2: {'time': 30}, 3: {'time': 30}, 4: {'time': 50}, 5: {'time': 70}, 6: {'time': 80}, 7: {'time': 90}, 8: {'time': 90}, 9: {'time': 80}, 10: {'time': 74}}
    nx.set_node_attributes(G, node_attrs)
    assert nx.cd_index(G, 4, time_delta=50) == 0.17