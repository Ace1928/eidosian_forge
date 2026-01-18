import pytest
import networkx as nx
def test_hide_multiedges(self):
    factory = nx.classes.filters.hide_multiedges
    f = factory([(1, 2, 0), (3, 4, 1), (1, 2, 1)])
    assert not f(1, 2, 0)
    assert not f(1, 2, 1)
    assert f(1, 2, 2)
    assert f(3, 4, 0)
    assert not f(3, 4, 1)
    assert not f(4, 3, 1)
    assert f(4, 3, 0)
    assert f(2, 3, 0)
    assert f(0, -1, 0)
    assert f('a', 'b', 0)
    pytest.raises(TypeError, f, 1, 2, 3, 4)
    pytest.raises(TypeError, f, 1, 2)
    pytest.raises(TypeError, f, 1)
    pytest.raises(TypeError, f)
    pytest.raises(TypeError, factory, [1, 2, 3])
    pytest.raises(ValueError, factory, [(1, 2)])
    pytest.raises(ValueError, factory, [(1, 2, 3, 4)])