import pytest
import networkx as nx
from networkx.utils import edges_equal
def same_attrdict(self, H, G):
    old_foo = H[1][2]['foo']
    H.edges[1, 2]['foo'] = 'baz'
    assert G.edges == H.edges
    H.edges[1, 2]['foo'] = old_foo
    assert G.edges == H.edges
    old_foo = H.nodes[0]['foo']
    H.nodes[0]['foo'] = 'baz'
    assert G.nodes == H.nodes
    H.nodes[0]['foo'] = old_foo
    assert G.nodes == H.nodes