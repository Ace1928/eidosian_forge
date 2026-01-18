import warnings
import pytest
import inspect
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.nanfunctions import _nan_mask, _replace_nan
from numpy.testing import (
@pytest.mark.parametrize('nanfunc,func', [(np.nanquantile, np.quantile), (np.nanpercentile, np.percentile)], ids=['nanquantile', 'nanpercentile'])
def test_nanfunc_q(self, mat, dtype, nanfunc, func):
    mat = mat.astype(dtype)
    if mat.dtype.kind == 'c':
        assert_raises(TypeError, func, mat, q=1)
        assert_raises(TypeError, nanfunc, mat, q=1)
    else:
        tgt = func(mat, q=1)
        out = nanfunc(mat, q=1)
        assert_almost_equal(out, tgt)
        if dtype == 'O':
            assert type(out) is type(tgt)
        else:
            assert out.dtype == tgt.dtype