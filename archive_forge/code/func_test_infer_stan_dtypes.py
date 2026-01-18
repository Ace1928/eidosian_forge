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
@pytest.mark.parametrize('model_code,expected', [('data {int y;} models {y ~ poisson(3);} generated quantities {int X;}', {'X': 'int'}), ('data {real y;} models {y ~ normal(0,1);} generated quantities {int Y; real G;}', {'Y': 'int'})])
def test_infer_stan_dtypes(model_code, expected):
    """Test different examples for dtypes in Stan models."""
    res = infer_stan_dtypes(model_code)
    assert res == expected