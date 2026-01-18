import math
from operator import itemgetter
import pytest
import networkx as nx
from networkx.algorithms.tree import branchings, recognition
def test_MultiDiGraph_EdgeKey():
    G = branchings.MultiDiGraph_EdgeKey()
    G.add_edge(1, 2, 'A')
    with pytest.raises(Exception, match="Key 'A' is already in use."):
        G.add_edge(3, 4, 'A')
    with pytest.raises(KeyError, match="Invalid edge key 'B'"):
        G.remove_edge_with_key('B')
    if G.remove_edge_with_key('A'):
        assert list(G.edges(data=True)) == []
    G.add_edge(1, 3, 'A')
    with pytest.raises(NotImplementedError):
        G.remove_edges_from([(1, 3)])