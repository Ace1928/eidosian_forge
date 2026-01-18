import pickle
import pytest
import networkx as nx
@pytest.mark.skipif("not nx._dispatch._automatic_backends or nx._dispatch._automatic_backends[0] != 'nx-loopback'")
def test_graph_converter_needs_backend():
    from networkx.classes.tests.dispatch_interface import LoopbackDispatcher, LoopbackGraph
    A = sp.sparse.coo_array([[0, 3, 2], [3, 0, 1], [2, 1, 0]])
    side_effects = []

    def from_scipy_sparse_array(self, *args, **kwargs):
        side_effects.append(1)
        return self.convert_from_nx(self.__getattr__('from_scipy_sparse_array')(*args, **kwargs), preserve_edge_attrs=None, preserve_node_attrs=None, preserve_graph_attrs=None)

    @staticmethod
    def convert_to_nx(obj, *, name=None):
        if type(obj) is nx.Graph:
            return obj
        return nx.Graph(obj)
    orig_convert_to_nx = LoopbackDispatcher.convert_to_nx
    LoopbackDispatcher.convert_to_nx = convert_to_nx
    LoopbackDispatcher.from_scipy_sparse_array = from_scipy_sparse_array
    try:
        assert side_effects == []
        assert type(nx.from_scipy_sparse_array(A)) is nx.Graph
        assert side_effects == [1]
        assert type(nx.from_scipy_sparse_array(A, backend='nx-loopback')) is LoopbackGraph
        assert side_effects == [1, 1]
    finally:
        LoopbackDispatcher.convert_to_nx = staticmethod(orig_convert_to_nx)
        del LoopbackDispatcher.from_scipy_sparse_array
    with pytest.raises(ImportError, match='Unable to load'):
        nx.from_scipy_sparse_array(A, backend='bad-backend-name')