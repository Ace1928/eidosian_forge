import scipy.linalg.interpolative as pymatrixid
import numpy as np
from scipy.linalg import hilbert, svdvals, norm
from scipy.sparse.linalg import aslinearoperator
from scipy.linalg.interpolative import interp_decomp
from numpy.testing import (assert_, assert_allclose, assert_equal,
import pytest
from pytest import raises as assert_raises
import sys
def test_id_to_svd(self, A, eps, rank):
    k = rank
    idx, proj = pymatrixid.interp_decomp(A, k, rand=False)
    U, S, V = pymatrixid.id_to_svd(A[:, idx[:k]], idx, proj)
    B = U * S @ V.T.conj()
    assert_allclose(A, B, rtol=eps, atol=1e-08)