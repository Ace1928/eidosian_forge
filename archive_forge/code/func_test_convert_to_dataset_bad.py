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
def test_convert_to_dataset_bad(tmpdir):
    first = convert_to_inference_data(np.random.randn(100), group='prior')
    filename = str(tmpdir.join('test_file.nc'))
    first.to_netcdf(filename)
    with pytest.raises(ValueError):
        convert_to_dataset(filename, group='bar')