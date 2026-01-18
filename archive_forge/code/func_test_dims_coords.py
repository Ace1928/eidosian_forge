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
def test_dims_coords():
    shape = (4, 20, 5)
    var_name = 'x'
    dims, coords = generate_dims_coords(shape, var_name)
    assert 'x_dim_0' in dims
    assert 'x_dim_1' in dims
    assert 'x_dim_2' in dims
    assert len(coords['x_dim_0']) == 4
    assert len(coords['x_dim_1']) == 20
    assert len(coords['x_dim_2']) == 5