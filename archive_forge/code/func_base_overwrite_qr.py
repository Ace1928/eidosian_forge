import itertools
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy import linalg
import scipy.linalg._decomp_update as _decomp_update
from scipy.linalg._decomp_update import qr_delete, qr_update, qr_insert
def base_overwrite_qr(self, which, p, test_C, test_F, mode='full'):
    assert_sqr = True if mode == 'full' else False
    if which == 'row':
        qind = (slice(p, None), slice(p, None))
        rind = (slice(p, None), slice(None))
    else:
        qind = (slice(None), slice(None))
        rind = (slice(None), slice(None, -p))
    a, q0, r0 = self.generate('sqr', mode)
    if p == 1:
        a1 = np.delete(a, 3, 0 if which == 'row' else 1)
    else:
        a1 = np.delete(a, slice(3, 3 + p), 0 if which == 'row' else 1)
    q = q0.copy('F')
    r = r0.copy('F')
    q1, r1 = qr_delete(q, r, 3, p, which, False)
    check_qr(q1, r1, a1, self.rtol, self.atol, assert_sqr)
    check_qr(q, r, a, self.rtol, self.atol, assert_sqr)
    if test_F:
        q = q0.copy('F')
        r = r0.copy('F')
        q2, r2 = qr_delete(q, r, 3, p, which, True)
        check_qr(q2, r2, a1, self.rtol, self.atol, assert_sqr)
        assert_allclose(q2, q[qind], rtol=self.rtol, atol=self.atol)
        assert_allclose(r2, r[rind], rtol=self.rtol, atol=self.atol)
    if test_C:
        q = q0.copy('C')
        r = r0.copy('C')
        q3, r3 = qr_delete(q, r, 3, p, which, True)
        check_qr(q3, r3, a1, self.rtol, self.atol, assert_sqr)
        assert_allclose(q3, q[qind], rtol=self.rtol, atol=self.atol)
        assert_allclose(r3, r[rind], rtol=self.rtol, atol=self.atol)