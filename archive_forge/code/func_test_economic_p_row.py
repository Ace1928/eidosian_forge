import itertools
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy import linalg
import scipy.linalg._decomp_update as _decomp_update
from scipy.linalg._decomp_update import qr_delete, qr_update, qr_insert
def test_economic_p_row(self):
    a, q, r, u = self.generate('tall', 'economic', 'row', 3)
    for row in range(r.shape[0] + 1):
        q1, r1 = qr_insert(q, r, u, row, overwrite_qru=False)
        a1 = np.insert(a, np.full(3, row, np.intp), u, 0)
        check_qr(q1, r1, a1, self.rtol, self.atol, False)