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
@pytest.mark.parametrize('nat', [np.datetime64('nat'), np.timedelta64('nat')])
def test_nat_is_not_inf(self, nat):
    try:
        assert not np.isinf(nat)
    except TypeError:
        pass