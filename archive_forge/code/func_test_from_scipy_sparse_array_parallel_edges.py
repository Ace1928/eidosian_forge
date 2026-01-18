import pytest
import networkx as nx
from networkx.generators.classic import barbell_graph, cycle_graph, path_graph
from networkx.utils import graphs_equal
def test_from_scipy_sparse_array_parallel_edges(self):
    """Tests that the :func:`networkx.from_scipy_sparse_array` function
        interprets integer weights as the number of parallel edges when
        creating a multigraph.

        """
    A = sp.sparse.csr_array([[1, 1], [1, 2]])
    expected = nx.DiGraph()
    edges = [(0, 0), (0, 1), (1, 0)]
    expected.add_weighted_edges_from([(u, v, 1) for u, v in edges])
    expected.add_edge(1, 1, weight=2)
    actual = nx.from_scipy_sparse_array(A, parallel_edges=True, create_using=nx.DiGraph)
    assert graphs_equal(actual, expected)
    actual = nx.from_scipy_sparse_array(A, parallel_edges=False, create_using=nx.DiGraph)
    assert graphs_equal(actual, expected)
    edges = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 1)]
    expected = nx.MultiDiGraph()
    expected.add_weighted_edges_from([(u, v, 1) for u, v in edges])
    actual = nx.from_scipy_sparse_array(A, parallel_edges=True, create_using=nx.MultiDiGraph)
    assert graphs_equal(actual, expected)
    expected = nx.MultiDiGraph()
    expected.add_edges_from(set(edges), weight=1)
    expected[1][1][0]['weight'] = 2
    actual = nx.from_scipy_sparse_array(A, parallel_edges=False, create_using=nx.MultiDiGraph)
    assert graphs_equal(actual, expected)