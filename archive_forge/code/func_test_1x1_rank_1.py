import itertools
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy import linalg
import scipy.linalg._decomp_update as _decomp_update
from scipy.linalg._decomp_update import qr_delete, qr_update, qr_insert
def test_1x1_rank_1(self):
    a, q, r, u, v = self.generate('1x1')
    q1, r1 = qr_update(q, r, u, v, False)
    a1 = a + np.outer(u, v.conj())
    check_qr(q1, r1, a1, self.rtol, self.atol)