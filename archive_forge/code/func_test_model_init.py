import threading
import time
from collections import Counter
import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu
from ..util import make_tempdir
def test_model_init():

    class MyShim(Shim):
        name = 'testshim'
    model_a = create_model('a')
    model = Model('test', lambda X: (X, lambda dY: dY), dims={'nI': 10, 'nO': None}, params={'W': numpy.zeros((10,)), 'b': None}, refs={'a': model_a, 'b': None}, attrs={'foo': 'bar'}, shims=[MyShim(None)], layers=[model_a, model_a])
    assert model.has_param('W')
    assert model.get_param('W').shape == (10,)
    assert model.has_param('b') is None
    with pytest.raises(KeyError):
        model.get_param('b')
    with pytest.raises(KeyError):
        model.get_param('X')
    model.set_param('X', numpy.zeros((10,)))
    assert model.has_param('X')
    assert model.get_param('X').shape == (10,)
    with model.use_params({(model.id, 'X'): numpy.ones((10,))}):
        assert numpy.array_equal(model.get_param('X'), numpy.ones((10,)))
    assert numpy.array_equal(model.get_param('X'), numpy.zeros((10,)))
    assert not model.has_grad('W')
    assert not model.has_grad('xyz')
    with pytest.raises(KeyError):
        model.get_grad('b')
    model.set_param('W', model.ops.alloc1f(10))
    model.set_grad('W', model.ops.alloc1f(10))
    with pytest.raises(ValueError):
        model.inc_grad('W', numpy.zeros((5, 0)))
    assert model.has_dim('nI')
    assert model.get_dim('nI') == 10
    with pytest.raises(KeyError):
        model.get_dim('xyz')
    with pytest.raises(ValueError):
        model.get_dim('nO')
    assert model.has_ref('a')
    assert model.get_ref('a').name == 'a'
    assert not model.has_ref('xyz')
    with pytest.raises(KeyError):
        model.get_ref('xyz')
    assert model.has_ref('b') is None
    with pytest.raises(ValueError):
        model.get_ref('b')
    model.set_ref('c', model_a)
    assert model.has_ref('c')
    assert model.get_ref('c').name == 'a'
    with pytest.raises(ValueError):
        model.set_ref('c', create_model('c'))
    assert 'foo' in model.attrs
    assert 'bar' not in model.attrs
    assert model.attrs['foo'] == 'bar'
    with pytest.raises(KeyError):
        model.attrs['bar']
    model.attrs['bar'] = 'baz'
    model_copy = model.copy()
    assert model_copy.name == 'test'