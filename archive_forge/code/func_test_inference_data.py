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
def test_inference_data(self, data, eight_schools_params):
    inference_data = self.get_inference_data(data, eight_schools_params)
    test_dict = {'posterior': [], 'prior': [], 'sample_stats': [], 'posterior_predictive': [], 'prior_predictive': [], 'sample_stats_prior': [], 'observed_data': ['J', 'y', 'sigma']}
    fails = check_multiple_attrs(test_dict, inference_data)
    assert not fails
    self.check_var_names_coords_dims(inference_data.posterior)
    self.check_var_names_coords_dims(inference_data.posterior_predictive)
    self.check_var_names_coords_dims(inference_data.sample_stats)
    self.check_var_names_coords_dims(inference_data.prior)
    self.check_var_names_coords_dims(inference_data.prior_predictive)
    self.check_var_names_coords_dims(inference_data.sample_stats_prior)
    pred_dims = inference_data.predictions.sizes['school_pred']
    assert pred_dims == 8