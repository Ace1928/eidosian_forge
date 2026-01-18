import threading
import time
from collections import Counter
import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu
from ..util import make_tempdir
def test_model_set_dim():

    class MyShim(Shim):
        name = 'testshim'
    model_a = create_model('a')
    model = Model('test', lambda X: (X, lambda dY: dY), dims={'nI': 5, 'nO': None}, params={'W': None, 'b': None}, refs={'a': model_a, 'b': None}, attrs={'foo': 'bar'}, shims=[MyShim(None)], layers=[model_a, model_a])
    with pytest.raises(ValueError):
        model.set_dim('nI', 10)
    model.set_dim('nI', 10, force=True)
    model.set_param('W', model.ops.alloc1f(10))
    model.set_grad('W', model.ops.alloc1f(10))
    assert model.has_dim('nI')
    assert model.get_dim('nI') == 10
    with pytest.raises(KeyError):
        model.set_dim('xyz', 20)
    with pytest.raises(ValueError):
        model.set_dim('nI', 20)
    with pytest.raises(ValueError):
        model.set_dim('nI', 20, force=True)