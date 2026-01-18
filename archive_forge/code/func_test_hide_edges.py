import pytest
import networkx as nx
def test_hide_edges(self):
    factory = nx.classes.filters.hide_edges
    f = factory([(1, 2), (3, 4)])
    assert not f(1, 2)
    assert not f(3, 4)
    assert not f(4, 3)
    assert f(2, 3)
    assert f(0, -1)
    assert f('a', 'b')
    pytest.raises(TypeError, f, 1, 2, 3)
    pytest.raises(TypeError, f, 1)
    pytest.raises(TypeError, f)
    pytest.raises(TypeError, factory, [1, 2, 3])
    pytest.raises(ValueError, factory, [(1, 2, 3)])