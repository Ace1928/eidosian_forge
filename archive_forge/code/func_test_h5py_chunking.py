import gc
import io
import random
import re
import string
import tempfile
from os import environ as env
import h5py
import netCDF4
import numpy as np
import pytest
from packaging import version
from pytest import raises
import h5netcdf
from h5netcdf import legacyapi
from h5netcdf.core import NOT_A_VARIABLE, CompatibilityError
def test_h5py_chunking(tmp_local_netcdf):
    with h5netcdf.File(tmp_local_netcdf, 'w') as ds:
        ds.dimensions = {'x': 10, 'y': 10, 'z': 10, 't': None}
        v = ds.create_variable('hello', ('x', 'y', 'z', 't'), 'float', chunking_heuristic='h5py')
        chunks_h5py = v.chunks
        ds.resize_dimension('t', 4)
        v = ds.create_variable('hello3', ('x', 'y', 'z', 't'), 'float', chunking_heuristic='h5py')
        chunks_resized = v.chunks
    with h5netcdf.File(tmp_local_netcdf, 'w') as ds:
        ds.dimensions = {'x': 10, 'y': 10, 'z': 10, 't': 1024}
        v = ds.create_variable('hello', ('x', 'y', 'z', 't'), 'float', chunks=True, chunking_heuristic='h5py')
        chunks_true = v.chunks
    with h5netcdf.File(tmp_local_netcdf, 'w') as ds:
        ds.dimensions = {'x': 10, 'y': 10, 'z': 10, 't': 4}
        v = ds.create_variable('hello', ('x', 'y', 'z', 't'), 'float', chunks=True, chunking_heuristic='h5py')
        chunks_true_resized = v.chunks
    assert chunks_h5py == chunks_true
    assert chunks_resized == chunks_true_resized