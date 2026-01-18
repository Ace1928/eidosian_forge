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
@pytest.mark.parametrize('engine,nc_format', ENGINES_AND_FORMATS)
def test_dask_distributed_netcdf_roundtrip(loop, tmp_netcdf_filename, engine, nc_format):
    if engine not in ENGINES:
        pytest.skip('engine not available')
    chunks = {'dim1': 4, 'dim2': 3, 'dim3': 6}
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop):
            original = create_test_data().chunk(chunks)
            if engine == 'scipy':
                with pytest.raises(NotImplementedError):
                    original.to_netcdf(tmp_netcdf_filename, engine=engine, format=nc_format)
                return
            original.to_netcdf(tmp_netcdf_filename, engine=engine, format=nc_format)
            with xr.open_dataset(tmp_netcdf_filename, chunks=chunks, engine=engine) as restored:
                assert isinstance(restored.var1.data, da.Array)
                computed = restored.compute()
                assert_allclose(original, computed)