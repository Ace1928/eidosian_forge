import scipy.linalg.interpolative as pymatrixid
import numpy as np
from scipy.linalg import hilbert, svdvals, norm
from scipy.sparse.linalg import aslinearoperator
from scipy.linalg.interpolative import interp_decomp
from numpy.testing import (assert_, assert_allclose, assert_equal,
import pytest
from pytest import raises as assert_raises
import sys
def test_estimate_spectral_norm(self, A):
    s = svdvals(A)
    norm_2_est = pymatrixid.estimate_spectral_norm(A)
    assert_allclose(norm_2_est, s[0], rtol=1e-06, atol=1e-08)