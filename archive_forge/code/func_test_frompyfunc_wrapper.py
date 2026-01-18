from __future__ import annotations
import pickle
import warnings
from functools import partial
from operator import add
import pytest
import dask.array as da
from dask.array.ufunc import da_frompyfunc
from dask.array.utils import assert_eq
from dask.base import tokenize
def test_frompyfunc_wrapper():
    f = da_frompyfunc(add, 2, 1)
    np_f = np.frompyfunc(add, 2, 1)
    x = np.array([1, 2, 3])
    np.testing.assert_equal(f(x, 1), np_f(x, 1))
    f2 = pickle.loads(pickle.dumps(f))
    np.testing.assert_equal(f2(x, 1), np_f(x, 1))
    assert f.ntypes == np_f.ntypes
    with pytest.raises(AttributeError):
        f.not_an_attribute
    assert 'ntypes' in dir(f)
    np.testing.assert_equal(f.outer(x, x), np_f.outer(x, x))
    assert f.__name__ == 'frompyfunc-add'
    assert repr(f) == 'da.frompyfunc<add, 2, 1>'
    assert tokenize(da_frompyfunc(add, 2, 1)) == tokenize(da_frompyfunc(add, 2, 1))