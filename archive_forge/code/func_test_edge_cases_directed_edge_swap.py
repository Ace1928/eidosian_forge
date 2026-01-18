import pytest
import networkx as nx
def test_edge_cases_directed_edge_swap():
    e = 'Maximum number of swap attempts \\(11\\) exceeded before desired swaps achieved \\(\\d\\).'
    graph = nx.DiGraph([(0, 0), (0, 1), (1, 0), (2, 3), (3, 2)])
    with pytest.raises(nx.NetworkXAlgorithmError, match=e):
        nx.directed_edge_swap(graph, nswap=1, max_tries=10, seed=1)