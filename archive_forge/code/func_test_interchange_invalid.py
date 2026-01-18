import itertools
import pytest
import networkx as nx
def test_interchange_invalid(self):
    graph = one_node_graph()
    for strategy in INTERCHANGE_INVALID:
        pytest.raises(nx.NetworkXPointlessConcept, nx.coloring.greedy_color, graph, strategy=strategy, interchange=True)