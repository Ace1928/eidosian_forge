import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_from_edgelist_multi_attr(self):
    Gtrue = nx.Graph([('E', 'C', {'cost': 9, 'weight': 10}), ('B', 'A', {'cost': 1, 'weight': 7}), ('A', 'D', {'cost': 7, 'weight': 4})])
    G = nx.from_pandas_edgelist(self.df, 0, 'b', ['weight', 'cost'])
    assert graphs_equal(G, Gtrue)