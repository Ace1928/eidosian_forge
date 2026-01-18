import os
import sys
import itertools
import traceback
import textwrap
import subprocess
import pytest
import numpy as np
from numpy import array, single, double, csingle, cdouble, dot, identity, matmul
from numpy.core import swapaxes
from numpy import multiply, atleast_2d, inf, asarray
from numpy import linalg
from numpy.linalg import matrix_power, norm, matrix_rank, multi_dot, LinAlgError
from numpy.linalg.linalg import _multi_dot_matrix_chain_order
from numpy.testing import (
def test_byteorder_check():
    if sys.byteorder == 'little':
        native = '<'
    else:
        native = '>'
    for dtt in (np.float32, np.float64):
        arr = np.eye(4, dtype=dtt)
        n_arr = arr.newbyteorder(native)
        sw_arr = arr.newbyteorder('S').byteswap()
        assert_equal(arr.dtype.byteorder, '=')
        for routine in (linalg.inv, linalg.det, linalg.pinv):
            res = routine(arr)
            assert_array_equal(res, routine(n_arr))
            assert_array_equal(res, routine(sw_arr))