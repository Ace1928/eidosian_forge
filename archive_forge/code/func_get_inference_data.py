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
def get_inference_data(self, data, eight_schools_params):
    return from_dict(posterior=data.obj, posterior_predictive=data.obj, sample_stats=data.obj, prior=data.obj, prior_predictive=data.obj, sample_stats_prior=data.obj, observed_data=eight_schools_params, coords={'school': np.array(['a' * i for i in range(8)], dtype='U')}, dims={'theta': ['school'], 'eta': ['school']})