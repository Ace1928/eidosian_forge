import pytest
import networkx as nx
from networkx.generators.classic import barbell_graph, cycle_graph, path_graph
from networkx.utils import graphs_equal
@pytest.mark.parametrize('sparse_format', ('csr', 'csc', 'dok'))
def test_from_scipy_sparse_array_formats(sparse_format):
    """Test all formats supported by _generate_weighted_edges."""
    expected = nx.Graph()
    expected.add_edges_from([(0, 1, {'weight': 3}), (0, 2, {'weight': 2}), (1, 0, {'weight': 3}), (1, 2, {'weight': 1}), (2, 0, {'weight': 2}), (2, 1, {'weight': 1})])
    A = sp.sparse.coo_array([[0, 3, 2], [3, 0, 1], [2, 1, 0]]).asformat(sparse_format)
    assert graphs_equal(expected, nx.from_scipy_sparse_array(A))