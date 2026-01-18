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
def test_exceptions_non_square(self, dt):
    assert_raises(LinAlgError, matrix_power, np.array([1], dt), 1)
    assert_raises(LinAlgError, matrix_power, np.array([[1], [2]], dt), 1)
    assert_raises(LinAlgError, matrix_power, np.ones((4, 3, 2), dt), 1)