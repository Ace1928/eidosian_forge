from itertools import combinations
import pytest
import networkx as nx
@pytest.mark.parametrize('graph', [path_graph(), fork_graph(), collider_graph(), naive_bayes_graph(), asia_graph()])
def test_markov_condition(graph):
    """Test that the Markov condition holds for each PGM graph."""
    for node in graph.nodes:
        parents = set(graph.predecessors(node))
        non_descendants = graph.nodes - nx.descendants(graph, node) - {node} - parents
        assert nx.d_separated(graph, {node}, non_descendants, parents)