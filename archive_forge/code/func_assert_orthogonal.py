import os
import pytest
import sys
import numpy as np
from numpy.testing import assert_allclose
from pytest import raises as assert_raises
from scipy.sparse.linalg._svdp import _svdp
from scipy.sparse import csr_matrix, csc_matrix
def assert_orthogonal(u1, u2, rtol, atol):
    """Check that the first k rows of u1 and u2 are orthogonal"""
    A = abs(np.dot(u1.conj().T, u2))
    assert_allclose(A, np.eye(u1.shape[1], u2.shape[1]), rtol=rtol, atol=atol)