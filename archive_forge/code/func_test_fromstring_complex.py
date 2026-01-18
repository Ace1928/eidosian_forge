import warnings
import platform
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.tests._locales import CommaDecimalPointLocale
def test_fromstring_complex():
    for ctype in ['complex', 'cdouble', 'cfloat']:
        assert_equal(np.fromstring('1, 2 ,  3  ,4', sep=',', dtype=ctype), np.array([1.0, 2.0, 3.0, 4.0]))
        assert_equal(np.fromstring('1j, -2j,  3j, 4e1j', sep=',', dtype=ctype), np.array([1j, -2j, 3j, 40j]))
        assert_equal(np.fromstring('1+1j,2-2j, -3+3j,  -4e1+4j', sep=',', dtype=ctype), np.array([1.0 + 1j, 2.0 - 2j, -3.0 + 3j, -40.0 + 4j]))
        with assert_warns(DeprecationWarning):
            assert_equal(np.fromstring('1+2 j,3', dtype=ctype, sep=','), np.array([1.0]))
        with assert_warns(DeprecationWarning):
            assert_equal(np.fromstring('1+ 2j,3', dtype=ctype, sep=','), np.array([1.0]))
        with assert_warns(DeprecationWarning):
            assert_equal(np.fromstring('1 +2j,3', dtype=ctype, sep=','), np.array([1.0]))
        with assert_warns(DeprecationWarning):
            assert_equal(np.fromstring('1+j', dtype=ctype, sep=','), np.array([1.0]))
        with assert_warns(DeprecationWarning):
            assert_equal(np.fromstring('1+', dtype=ctype, sep=','), np.array([1.0]))
        with assert_warns(DeprecationWarning):
            assert_equal(np.fromstring('1j+1', dtype=ctype, sep=','), np.array([1j]))