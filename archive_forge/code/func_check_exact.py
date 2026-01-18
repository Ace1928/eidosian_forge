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
def check_exact(self, a, info):
    mmwrite(self.fn, a)
    assert_equal(mminfo(self.fn), info)
    b = mmread(self.fn)
    assert_equal(a.toarray(), b.toarray())