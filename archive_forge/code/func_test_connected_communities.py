from itertools import chain, combinations
import pytest
import networkx as nx
def test_connected_communities():
    test = nx.Graph()
    test.add_edge('a', 'b')
    test.add_edge('c', 'a')
    test.add_edge('c', 'b')
    test.add_edge('d', 'a')
    test.add_edge('d', 'b')
    test.add_edge('d', 'c')
    test.add_edge('e', 'a')
    test.add_edge('e', 'b')
    test.add_edge('e', 'c')
    test.add_edge('e', 'd')
    test.add_edge('1', '2')
    test.add_edge('3', '1')
    test.add_edge('3', '2')
    test.add_edge('4', '1')
    test.add_edge('4', '2')
    test.add_edge('4', '3')
    test.add_edge('5', '1')
    test.add_edge('5', '2')
    test.add_edge('5', '3')
    test.add_edge('5', '4')
    test.add_edge('a', '1')
    test.add_edge('x', 'y')
    test.add_node('z')
    ground_truth1 = {frozenset(['a', 'b', 'c', 'd', 'e']), frozenset(['1', '2', '3', '4', '5']), frozenset(['x', 'y']), frozenset(['z'])}
    ground_truth2 = {frozenset(['a', 'b', 'c', 'd', 'e', '1', '2', '3', '4', '5']), frozenset(['x', 'y']), frozenset(['z'])}
    ground_truth = (ground_truth1, ground_truth2)
    communities = nx.community.label_propagation_communities(test)
    result = {frozenset(c) for c in communities}
    assert result in ground_truth