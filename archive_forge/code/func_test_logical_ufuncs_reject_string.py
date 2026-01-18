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
@pytest.mark.parametrize('ufunc', [np.logical_and, np.logical_or, np.logical_xor])
def test_logical_ufuncs_reject_string(self, ufunc):
    """
        Logical ufuncs are normally well defined by working with the boolean
        equivalent, i.e. casting all inputs to bools should work.

        However, casting strings to bools is *currently* weird, because it
        actually uses `bool(int(str))`.  Thus we explicitly reject strings.
        This test should succeed (and can probably just be removed) as soon as
        string to bool casts are well defined in NumPy.
        """
    with pytest.raises(TypeError, match='contain a loop with signature'):
        ufunc(['1'], ['3'])
    with pytest.raises(TypeError, match='contain a loop with signature'):
        ufunc.reduce(['1', '2', '0'])