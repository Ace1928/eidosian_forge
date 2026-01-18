import sys
import math
import numpy as np
from numpy import sqrt, cos, sin, arctan, exp, log, pi
from numpy.testing import (assert_,
import pytest
from scipy.integrate import quad, dblquad, tplquad, nquad
from scipy.special import erf, erfc
from scipy._lib._ccallback import LowLevelCallable
import ctypes
import ctypes.util
from scipy._lib._ccallback_c import sine_ctypes
import scipy.integrate._test_multivariate as clib_test
def test_ctypes_variants(self):
    sin_0 = get_clib_test_routine('_sin_0', ctypes.c_double, ctypes.c_double, ctypes.c_void_p)
    sin_1 = get_clib_test_routine('_sin_1', ctypes.c_double, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)
    sin_2 = get_clib_test_routine('_sin_2', ctypes.c_double, ctypes.c_double)
    sin_3 = get_clib_test_routine('_sin_3', ctypes.c_double, ctypes.c_int, ctypes.POINTER(ctypes.c_double))
    sin_4 = get_clib_test_routine('_sin_3', ctypes.c_double, ctypes.c_int, ctypes.c_double)
    all_sigs = [sin_0, sin_1, sin_2, sin_3, sin_4]
    legacy_sigs = [sin_2, sin_4]
    legacy_only_sigs = [sin_4]
    for j, func in enumerate(all_sigs):
        callback = LowLevelCallable(func)
        if func in legacy_only_sigs:
            pytest.raises(ValueError, quad, callback, 0, pi)
        else:
            assert_allclose(quad(callback, 0, pi)[0], 2.0)
    for j, func in enumerate(legacy_sigs):
        if func in legacy_sigs:
            assert_allclose(quad(func, 0, pi)[0], 2.0)
        else:
            pytest.raises(ValueError, quad, func, 0, pi)