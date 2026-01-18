import importlib
import os
from collections import namedtuple
from copy import deepcopy
from html import escape
from typing import Dict
from tempfile import TemporaryDirectory
from urllib.parse import urlunsplit
import numpy as np
import pytest
import xarray as xr
from xarray.core.options import OPTIONS
from xarray.testing import assert_identical
from ... import (
from ...data.base import dict_to_dataset, generate_dims_coords, infer_stan_dtypes, make_attrs
from ...data.datasets import LOCAL_DATASETS, REMOTE_DATASETS, RemoteFileMetadata
from ..helpers import (  # pylint: disable=unused-import
@pytest.mark.parametrize('fill_attrs', [True, False])
def test_local_save(fill_attrs):
    inference_data = load_arviz_data('centered_eight')
    assert isinstance(inference_data, InferenceData)
    if fill_attrs:
        inference_data.attrs['test'] = 1
    with TemporaryDirectory(prefix='arviz_tests_') as tmp_dir:
        path = os.path.join(tmp_dir, 'test_file.nc')
        inference_data.to_netcdf(path)
        inference_data2 = from_netcdf(path)
        if fill_attrs:
            assert 'test' in inference_data2.attrs
            assert inference_data2.attrs['test'] == 1
        assert all((group in inference_data2 for group in inference_data._groups_all))