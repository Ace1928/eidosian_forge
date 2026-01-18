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
@pytest.mark.parametrize('base_group', ['/', 'test_group', 'group/subgroup'])
@pytest.mark.parametrize('groups_arg', [False, True])
@pytest.mark.parametrize('compress', [True, False])
@pytest.mark.parametrize('engine', ['h5netcdf', 'netcdf4'])
def test_io_method(self, data, eight_schools_params, groups_arg, base_group, compress, engine):
    inference_data = self.get_inference_data(data, eight_schools_params)
    if engine == 'h5netcdf':
        try:
            import h5netcdf
        except ImportError:
            pytest.skip('h5netcdf not installed')
    elif engine == 'netcdf4':
        try:
            import netCDF4
        except ImportError:
            pytest.skip('netcdf4 not installed')
    test_dict = {'posterior': ['eta', 'theta', 'mu', 'tau'], 'posterior_predictive': ['eta', 'theta', 'mu', 'tau'], 'sample_stats': ['eta', 'theta', 'mu', 'tau'], 'prior': ['eta', 'theta', 'mu', 'tau'], 'prior_predictive': ['eta', 'theta', 'mu', 'tau'], 'sample_stats_prior': ['eta', 'theta', 'mu', 'tau'], 'observed_data': ['J', 'y', 'sigma']}
    fails = check_multiple_attrs(test_dict, inference_data)
    assert not fails
    here = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(here, '..', 'saved_models')
    filepath = os.path.join(data_directory, 'io_method_testfile.nc')
    assert not os.path.exists(filepath)
    inference_data.to_netcdf(filepath, groups=('posterior', 'observed_data') if groups_arg else None, compress=compress, base_group=base_group)
    assert os.path.exists(filepath)
    assert os.path.getsize(filepath) > 0
    inference_data2 = InferenceData.from_netcdf(filepath, base_group=base_group)
    if groups_arg:
        test_dict = {'posterior': ['eta', 'theta', 'mu', 'tau'], 'observed_data': ['J', 'y', 'sigma']}
        assert not hasattr(inference_data2, 'sample_stats')
    fails = check_multiple_attrs(test_dict, inference_data2)
    assert not fails
    os.remove(filepath)
    assert not os.path.exists(filepath)