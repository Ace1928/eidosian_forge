from __future__ import annotations
import sys
from importlib.metadata import EntryPoint
from unittest import mock
import pytest
from xarray.backends import common, plugins
from xarray.tests import (
importlib_metadata_mock = "importlib.metadata"
def test_list_engines() -> None:
    from xarray.backends import list_engines
    engines = list_engines()
    assert list_engines.cache_info().currsize == 1
    assert ('scipy' in engines) == has_scipy
    assert ('h5netcdf' in engines) == has_h5netcdf
    assert ('netcdf4' in engines) == has_netCDF4
    assert ('pydap' in engines) == has_pydap
    assert ('zarr' in engines) == has_zarr
    assert ('pynio' in engines) == has_pynio
    assert 'store' in engines