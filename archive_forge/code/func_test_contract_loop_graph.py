import pytest
import networkx as nx
from networkx.utils import arbitrary_element, edges_equal, nodes_equal
def test_contract_loop_graph():
    """Tests for node contraction when nodes have loops."""
    G = nx.cycle_graph(4)
    G.add_edge(0, 0)
    actual = nx.contracted_nodes(G, 0, 1)
    expected = nx.complete_graph([0, 2, 3])
    expected.add_edge(0, 0)
    expected.add_edge(0, 0)
    assert edges_equal(actual.edges, expected.edges)
    actual = nx.contracted_nodes(G, 1, 0)
    expected = nx.complete_graph([1, 2, 3])
    expected.add_edge(1, 1)
    expected.add_edge(1, 1)
    assert edges_equal(actual.edges, expected.edges)