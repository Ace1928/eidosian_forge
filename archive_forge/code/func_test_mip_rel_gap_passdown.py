import sys
import platform
import numpy as np
from numpy.testing import (assert_, assert_allclose, assert_equal,
from pytest import raises as assert_raises
from scipy.optimize import linprog, OptimizeWarning
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse.linalg import MatrixRankWarning
from scipy.linalg import LinAlgWarning
from scipy._lib._util import VisibleDeprecationWarning
import scipy.sparse
import pytest
@pytest.mark.xslow
def test_mip_rel_gap_passdown(self):
    A_eq = np.array([[22, 13, 26, 33, 21, 3, 14, 26], [39, 16, 22, 28, 26, 30, 23, 24], [18, 14, 29, 27, 30, 38, 26, 26], [41, 26, 28, 36, 18, 38, 16, 26]])
    b_eq = np.array([7872, 10466, 11322, 12058])
    c = np.array([2, 10, 13, 17, 7, 5, 7, 3])
    bounds = [(0, np.inf)] * 8
    integrality = [1] * 8
    mip_rel_gaps = [0.5, 0.25, 0.01, 0.001]
    sol_mip_gaps = []
    for mip_rel_gap in mip_rel_gaps:
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=self.method, integrality=integrality, options={'mip_rel_gap': mip_rel_gap})
        final_mip_gap = res['mip_gap']
        assert final_mip_gap <= mip_rel_gap
        sol_mip_gaps.append(final_mip_gap)
    gap_diffs = np.diff(np.flip(sol_mip_gaps))
    assert np.all(gap_diffs >= 0)
    assert not np.all(gap_diffs == 0)