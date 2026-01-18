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
def test_mode_raw(self):
    a = self.array([[1, 2], [3, 4], [5, 6]], dtype=np.double)
    h, tau = linalg.qr(a, mode='raw')
    assert_(h.dtype == np.double)
    assert_(tau.dtype == np.double)
    assert_(h.shape == (2, 3))
    assert_(tau.shape == (2,))
    h, tau = linalg.qr(a.T, mode='raw')
    assert_(h.dtype == np.double)
    assert_(tau.dtype == np.double)
    assert_(h.shape == (3, 2))
    assert_(tau.shape == (2,))