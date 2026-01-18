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
def test_large_power(self, dt):
    rshft = self.rshft_1.astype(dt)
    assert_equal(matrix_power(rshft, 2 ** 100 + 2 ** 10 + 2 ** 5 + 0), self.rshft_0)
    assert_equal(matrix_power(rshft, 2 ** 100 + 2 ** 10 + 2 ** 5 + 1), self.rshft_1)
    assert_equal(matrix_power(rshft, 2 ** 100 + 2 ** 10 + 2 ** 5 + 2), self.rshft_2)
    assert_equal(matrix_power(rshft, 2 ** 100 + 2 ** 10 + 2 ** 5 + 3), self.rshft_3)