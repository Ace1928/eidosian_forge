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
def test_mapwrapper_serial():
    in_arg = np.arange(10.0)
    out_arg = np.sin(in_arg)
    p = MapWrapper(1)
    assert_(p._mapfunc is map)
    assert_(p.pool is None)
    assert_(p._own_pool is False)
    out = list(p(np.sin, in_arg))
    assert_equal(out, out_arg)
    with assert_raises(RuntimeError):
        p = MapWrapper(0)