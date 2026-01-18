import warnings
import itertools
import sys
import ctypes as ct
import pytest
from pytest import param
import numpy as np
import numpy.core._umath_tests as umt
import numpy.linalg._umath_linalg as uml
import numpy.core._operand_flag_tests as opflag_tests
import numpy.core._rational_tests as _rational_tests
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.compat import pickle
@pytest.mark.parametrize('ufunc', [np.logical_and, np.logical_or])
@pytest.mark.parametrize('signature', [(None, None, object), (object, None, None), (None, object, None)])
def test_logical_ufuncs_object_signatures(self, ufunc, signature):
    a = np.array([True, None, False], dtype=object)
    res = ufunc(a, a, signature=signature)
    assert res.dtype == object