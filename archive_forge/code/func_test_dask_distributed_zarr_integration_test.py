from __future__ import annotations
import pickle
from typing import TYPE_CHECKING, Any
import numpy as np
import pytest
from dask.distributed import Client, Lock
from distributed.client import futures_of
from distributed.utils_test import (  # noqa: F401
import xarray as xr
from xarray.backends.locks import HDF5_LOCK, CombinedLock, SerializableLock
from xarray.tests import (
from xarray.tests.test_backends import (
from xarray.tests.test_dataset import create_test_data
@requires_zarr
@pytest.mark.parametrize('consolidated', [True, False])
@pytest.mark.parametrize('compute', [True, False])
def test_dask_distributed_zarr_integration_test(loop, consolidated: bool, compute: bool) -> None:
    if consolidated:
        write_kwargs: dict[str, Any] = {'consolidated': True}
        read_kwargs: dict[str, Any] = {'backend_kwargs': {'consolidated': True}}
    else:
        write_kwargs = read_kwargs = {}
    chunks = {'dim1': 4, 'dim2': 3, 'dim3': 5}
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop):
            original = create_test_data().chunk(chunks)
            with create_tmp_file(allow_cleanup_failure=ON_WINDOWS, suffix='.zarrc') as filename:
                maybe_futures = original.to_zarr(filename, compute=compute, **write_kwargs)
                if not compute:
                    maybe_futures.compute()
                with xr.open_dataset(filename, chunks='auto', engine='zarr', **read_kwargs) as restored:
                    assert isinstance(restored.var1.data, da.Array)
                    computed = restored.compute()
                    assert_allclose(original, computed)