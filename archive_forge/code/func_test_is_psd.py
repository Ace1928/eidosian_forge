import numpy as np
import scipy.sparse.linalg as sparla
import cvxpy as cp
import cvxpy.settings as s
from cvxpy import psd_wrap
def test_is_psd() -> None:
    n = 50
    psd = np.eye(n)
    nsd = -np.eye(n)
    assert cp.Constant(psd).is_psd()
    assert not cp.Constant(psd).is_nsd()
    assert cp.Constant(nsd).is_nsd()
    assert not cp.Constant(nsd).is_psd()
    failures = set()
    for seed in range(95, 100):
        np.random.seed(seed)
        P = np.random.randn(n, n)
        P = P.T @ P
        try:
            cp.Constant(P).is_psd()
        except sparla.ArpackNoConvergence as e:
            assert 'CVXPY note' in str(e)
            failures.add(seed)
    assert failures == {97}
    assert psd_wrap(cp.Constant(P)).is_psd()