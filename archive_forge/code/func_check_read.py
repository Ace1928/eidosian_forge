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
def check_read(self, example, a, info):
    f = open(self.fn, 'w')
    f.write(example)
    f.close()
    assert_equal(mminfo(self.fn), info)
    b = mmread(self.fn).toarray()
    assert_array_almost_equal(a, b)