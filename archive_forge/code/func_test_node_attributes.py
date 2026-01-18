import pytest
import networkx as nx
from networkx.utils import arbitrary_element, edges_equal, nodes_equal
def test_node_attributes():
    """Tests that node contraction preserves node attributes."""
    G = nx.cycle_graph(4)
    G.nodes[0]['foo'] = 'bar'
    G.nodes[1]['baz'] = 'xyzzy'
    actual = nx.contracted_nodes(G, 0, 1)
    expected = nx.complete_graph(3)
    expected = nx.relabel_nodes(expected, {1: 2, 2: 3})
    expected.add_edge(0, 0)
    cdict = {1: {'baz': 'xyzzy'}}
    expected.nodes[0].update({'foo': 'bar', 'contraction': cdict})
    assert nx.is_isomorphic(actual, expected)
    assert actual.nodes == expected.nodes