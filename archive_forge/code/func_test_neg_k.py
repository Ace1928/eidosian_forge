import itertools
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy import linalg
import scipy.linalg._decomp_update as _decomp_update
from scipy.linalg._decomp_update import qr_delete, qr_update, qr_insert
def test_neg_k(self):
    a, q, r = self.generate('sqr')
    for k, p, w in itertools.product([-3, -7], [1, 3], ['row', 'col']):
        q1, r1 = qr_delete(q, r, k, p, w, overwrite_qr=False)
        if w == 'row':
            a1 = np.delete(a, slice(k + a.shape[0], k + p + a.shape[0]), 0)
        else:
            a1 = np.delete(a, slice(k + a.shape[0], k + p + a.shape[1]), 1)
        check_qr(q1, r1, a1, self.rtol, self.atol)