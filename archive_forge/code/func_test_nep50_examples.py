import operator
import numpy as np
import pytest
from numpy.testing import IS_WASM
@pytest.mark.skipif(IS_WASM, reason="wasm doesn't have support for fp errors")
def test_nep50_examples():
    with pytest.warns(UserWarning, match='result dtype changed'):
        res = np.uint8(1) + 2
    assert res.dtype == np.uint8
    with pytest.warns(UserWarning, match='result dtype changed'):
        res = np.array([1], np.uint8) + np.int64(1)
    assert res.dtype == np.int64
    with pytest.warns(UserWarning, match='result dtype changed'):
        res = np.array([1], np.uint8) + np.array(1, dtype=np.int64)
    assert res.dtype == np.int64
    with pytest.warns(UserWarning, match='result dtype changed'):
        with np.errstate(over='raise'):
            res = np.uint8(100) + 200
    assert res.dtype == np.uint8
    with pytest.warns(Warning) as recwarn:
        res = np.float32(1) + 3e+100
    warning = str(recwarn.pop(UserWarning).message)
    assert warning.startswith('result dtype changed')
    warning = str(recwarn.pop(RuntimeWarning).message)
    assert warning.startswith('overflow')
    assert len(recwarn) == 0
    assert np.isinf(res)
    assert res.dtype == np.float32
    res = np.array([0.1], np.float32) == np.float64(0.1)
    assert res[0] == False
    with pytest.warns(UserWarning, match='result dtype changed'):
        res = np.array([0.1], np.float32) + np.float64(0.1)
    assert res.dtype == np.float64
    with pytest.warns(UserWarning, match='result dtype changed'):
        res = np.array([1.0], np.float32) + np.int64(3)
    assert res.dtype == np.float64