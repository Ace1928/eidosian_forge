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
def test__get_strided_loop_errors_bad_call_info(self):
    i4 = np.dtype('i4')
    dt, call_info = np.negative._resolve_dtypes_and_context((i4, i4))
    with pytest.raises(ValueError, match='PyCapsule'):
        np.negative._get_strided_loop('not the capsule!')
    with pytest.raises(TypeError, match='.*incompatible context'):
        np.add._get_strided_loop(call_info)
    np.negative._get_strided_loop(call_info)
    with pytest.raises(TypeError):
        np.negative._get_strided_loop(call_info)