from __future__ import annotations
import sys
import pytest
import numpy as np
import scipy.linalg
from packaging.version import parse as parse_version
import dask.array as da
from dask.array.linalg import qr, sfqr, svd, svd_compressed, tsqr
from dask.array.numpy_compat import _np_version
from dask.array.utils import assert_eq, same_keys, svd_flip
@pytest.mark.parametrize('m,n', [(10, 20), (15, 15), (20, 10)])
def test_dask_svd_self_consistent(m, n):
    a = np.random.default_rng().random((m, n))
    d_a = da.from_array(a, chunks=(3, n), name='A')
    d_u, d_s, d_vt = da.linalg.svd(d_a)
    u, s, vt = da.compute(d_u, d_s, d_vt)
    for d_e, e in zip([d_u, d_s, d_vt], [u, s, vt]):
        assert d_e.shape == e.shape
        assert d_e.dtype == e.dtype