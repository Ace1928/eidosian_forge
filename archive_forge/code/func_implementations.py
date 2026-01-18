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
@pytest.fixture(scope='module', params=(scipy.io._mmio, fmm), autouse=True)
def implementations(request):
    global mminfo
    global mmread
    global mmwrite
    mminfo = request.param.mminfo
    mmread = request.param.mmread
    mmwrite = request.param.mmwrite