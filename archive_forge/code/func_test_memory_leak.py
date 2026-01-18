import gc
import pickle
import platform
import weakref
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
@pytest.mark.skipif(platform.python_implementation() == 'PyPy', reason='PyPy gc is different')
def test_memory_leak(self):
    G = self.Graph()

    def count_objects_of_type(_type):
        return sum((1 for obj in gc.get_objects() if not isinstance(obj, weakref.ProxyTypes) and isinstance(obj, _type)))
    gc.collect()
    before = count_objects_of_type(self.Graph)
    G.copy()
    gc.collect()
    after = count_objects_of_type(self.Graph)
    assert before == after

    class MyGraph(self.Graph):
        pass
    gc.collect()
    G = MyGraph()
    before = count_objects_of_type(MyGraph)
    G.copy()
    gc.collect()
    after = count_objects_of_type(MyGraph)
    assert before == after