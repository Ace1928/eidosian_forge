import itertools
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy import linalg
import scipy.linalg._decomp_update as _decomp_update
from scipy.linalg._decomp_update import qr_delete, qr_update, qr_insert
def test_Mx1_rank_p(self):
    a, q, r, u, v = self.generate('Mx1', p=1)
    u = u.reshape(u.size, 1)
    v = v.reshape(v.size, 1)
    q1, r1 = qr_update(q, r, u, v, False)
    a1 = a + np.dot(u, v.T.conj())
    check_qr(q1, r1, a1, self.rtol, self.atol)