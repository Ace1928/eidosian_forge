from __future__ import annotations
import inspect
import pytest
import dask
from dask import delayed
from dask.base import collections_to_dsk, key_split, visualize_dsk
from dask.core import get_deps
from dask.order import _connecting_to_roots, diagnostics, ndependencies, order
from dask.utils_test import add, inc
@pytest.mark.parametrize('optimize', [True, False])
def test_array_vs_dataframe(optimize):
    xr = pytest.importorskip('xarray')
    dd = pytest.importorskip('dask.dataframe')
    if dd._dask_expr_enabled():
        pytest.xfail("doesn't work yet")
    import dask.array as da
    size = 5000
    ds = xr.Dataset(dict(anom_u=(['time', 'face', 'j', 'i'], da.random.random((size, 1, 987, 1920), chunks=(10, 1, -1, -1))), anom_v=(['time', 'face', 'j', 'i'], da.random.random((size, 1, 987, 1920), chunks=(10, 1, -1, -1)))))
    quad = ds ** 2
    quad['uv'] = ds.anom_u * ds.anom_v
    mean = quad.mean('time')
    diag_array = diagnostics(collections_to_dsk([mean], optimize_graph=optimize))
    diag_df = diagnostics(collections_to_dsk([mean.to_dask_dataframe()], optimize_graph=optimize))
    assert max(diag_df[1]) == max(diag_array[1])
    assert max(diag_array[1]) < 50