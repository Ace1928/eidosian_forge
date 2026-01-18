import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_from_edgelist_multi_attr_incl_target(self):
    Gtrue = nx.Graph([('E', 'C', {0: 'C', 'b': 'E', 'weight': 10}), ('B', 'A', {0: 'B', 'b': 'A', 'weight': 7}), ('A', 'D', {0: 'A', 'b': 'D', 'weight': 4})])
    G = nx.from_pandas_edgelist(self.df, 0, 'b', [0, 'b', 'weight'])
    assert graphs_equal(G, Gtrue)