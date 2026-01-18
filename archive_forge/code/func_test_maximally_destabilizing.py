from datetime import datetime, timedelta
import pytest
import networkx as nx
def test_maximally_destabilizing(self):
    G = nx.DiGraph()
    G.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    G.add_edge(5, 1)
    G.add_edge(5, 2)
    G.add_edge(5, 3)
    G.add_edge(5, 4)
    G.add_edge(6, 5)
    G.add_edge(7, 5)
    G.add_edge(8, 5)
    G.add_edge(9, 5)
    G.add_edge(10, 5)
    G.add_edge(11, 5)
    node_attrs = {0: {'time': datetime(1992, 1, 1)}, 1: {'time': datetime(1992, 1, 1)}, 2: {'time': datetime(1993, 1, 1)}, 3: {'time': datetime(1993, 1, 1)}, 4: {'time': datetime(1995, 1, 1)}, 5: {'time': datetime(1997, 1, 1)}, 6: {'time': datetime(1998, 1, 1)}, 7: {'time': datetime(1999, 1, 1)}, 8: {'time': datetime(1999, 1, 1)}, 9: {'time': datetime(1998, 1, 1)}, 10: {'time': datetime(1997, 4, 1)}, 11: {'time': datetime(1998, 5, 1)}}
    nx.set_node_attributes(G, node_attrs)
    assert nx.cd_index(G, 5, time_delta=_delta) == 1