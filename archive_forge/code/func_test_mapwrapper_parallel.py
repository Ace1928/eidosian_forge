from multiprocessing import Pool
from multiprocessing.pool import Pool as PWL
import re
import math
from fractions import Fraction
import numpy as np
from numpy.testing import assert_equal, assert_
import pytest
from pytest import raises as assert_raises
import hypothesis.extra.numpy as npst
from hypothesis import given, strategies, reproduce_failure  # noqa: F401
from scipy.conftest import array_api_compatible
from scipy._lib._array_api import xp_assert_equal
from scipy._lib._util import (_aligned_zeros, check_random_state, MapWrapper,
def test_mapwrapper_parallel():
    in_arg = np.arange(10.0)
    out_arg = np.sin(in_arg)
    with MapWrapper(2) as p:
        out = p(np.sin, in_arg)
        assert_equal(list(out), out_arg)
        assert_(p._own_pool is True)
        assert_(isinstance(p.pool, PWL))
        assert_(p._mapfunc is not None)
    with assert_raises(Exception) as excinfo:
        p(np.sin, in_arg)
    assert_(excinfo.type is ValueError)
    with Pool(2) as p:
        q = MapWrapper(p.map)
        assert_(q._own_pool is False)
        q.close()
        out = p.map(np.sin, in_arg)
        assert_equal(list(out), out_arg)