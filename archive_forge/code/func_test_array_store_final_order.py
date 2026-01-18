from __future__ import annotations
import inspect
import pytest
import dask
from dask import delayed
from dask.base import collections_to_dsk, key_split, visualize_dsk
from dask.core import get_deps
from dask.order import _connecting_to_roots, diagnostics, ndependencies, order
from dask.utils_test import add, inc
def test_array_store_final_order(tmpdir):
    da = pytest.importorskip('dask.array')
    zarr = pytest.importorskip('zarr')
    arrays = [da.ones((110, 4), chunks=(100, 2)) for i in range(4)]
    x = da.concatenate(arrays, axis=0).rechunk((100, 2))
    store = zarr.DirectoryStore(tmpdir)
    root = zarr.group(store, overwrite=True)
    dest = root.empty_like(name='dest', data=x, chunks=x.chunksize, overwrite=True)
    d = x.store(dest, lock=False, compute=False)
    o = order(d.dask)
    assert_topological_sort(dict(d.dask), o)
    stores = [k for k in o if isinstance(k, tuple) and k[0].startswith('store-map-')]
    first_store = min(stores, key=lambda k: o[k])
    connected_stores = [k for k in stores if k[-1] == first_store[-1]]
    disconnected_stores = [k for k in stores if k[-1] != first_store[-1]]
    connected_max = max((v for k, v in o.items() if k in connected_stores))
    disconnected_min = min((v for k, v in o.items() if k in disconnected_stores))
    assert connected_max < disconnected_min