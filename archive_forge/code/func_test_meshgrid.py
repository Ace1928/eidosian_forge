from __future__ import annotations
import pytest
import numpy as np
import pytest
from tlz import concat
import dask
import dask.array as da
from dask.array.core import normalize_chunks
from dask.array.numpy_compat import AxisError
from dask.array.utils import assert_eq, same_keys
@pytest.mark.parametrize('shapes, chunks', [([()], [()]), ([(0,)], [(0,)]), ([(2,), (3,)], [(1,), (2,)]), ([(2,), (3,), (4,)], [(1,), (2,), (3,)]), ([(2,), (3,), (4,), (5,)], [(1,), (2,), (3,), (4,)]), ([(2, 3), (4,)], [(1, 2), (3,)])])
@pytest.mark.parametrize('indexing', ['ij', 'xy'])
@pytest.mark.parametrize('sparse', [False, True])
def test_meshgrid(shapes, chunks, indexing, sparse):
    xi_a = []
    xi_d = []
    xi_dc = []
    for each_shape, each_chunk in zip(shapes, chunks):
        xi_a.append(np.random.random(each_shape))
        xi_d_e = da.from_array(xi_a[-1], chunks=each_chunk)
        xi_d.append(xi_d_e)
        xi_d_ef = xi_d_e.flatten()
        xi_dc.append(xi_d_ef.chunks[0])
    do = list(range(len(xi_dc)))
    if indexing == 'xy' and len(xi_dc) > 1:
        do[0], do[1] = (do[1], do[0])
        xi_dc[0], xi_dc[1] = (xi_dc[1], xi_dc[0])
    xi_dc = tuple(xi_dc)
    r_a = np.meshgrid(*xi_a, indexing=indexing, sparse=sparse)
    r_d = da.meshgrid(*xi_d, indexing=indexing, sparse=sparse)
    assert isinstance(r_d, list)
    assert len(r_a) == len(r_d)
    for e_r_a, e_r_d, i in zip(r_a, r_d, do):
        assert_eq(e_r_a, e_r_d)
        if sparse:
            assert e_r_d.chunks[i] == xi_dc[i]
        else:
            assert e_r_d.chunks == xi_dc