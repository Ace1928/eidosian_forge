from tempfile import mkdtemp
import os
import io
import shutil
import textwrap
import numpy as np
from numpy import array, transpose, pi
from numpy.testing import (assert_equal, assert_allclose,
import pytest
from pytest import raises as assert_raises
import scipy.sparse
import scipy.io._mmio
import scipy.io._fast_matrix_market as fmm
def test_gh11389():
    mmread(io.StringIO('%%MatrixMarket matrix coordinate complex symmetric\n 1 1 1\n1 1 -2.1846000000000e+02  0.0000000000000e+00'))