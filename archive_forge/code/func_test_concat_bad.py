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
def test_concat_bad():
    with pytest.raises(TypeError):
        concat('hello', 'hello')
    idata = from_dict(posterior={'A': np.random.randn(2, 10, 2), 'B': np.random.randn(2, 10, 5, 2)})
    idata2 = from_dict(posterior={'A': np.random.randn(2, 10, 2)})
    idata3 = from_dict(prior={'A': np.random.randn(2, 10, 2)})
    with pytest.raises(TypeError):
        concat(idata, np.array([1, 2, 3, 4, 5]))
    with pytest.raises(TypeError):
        concat(idata, idata, dim=None)
    with pytest.raises(TypeError):
        concat(idata, idata2, dim='chain')
    with pytest.raises(TypeError):
        concat(idata2, idata, dim='chain')
    with pytest.raises(TypeError):
        concat(idata, idata3, dim='chain')
    with pytest.raises(TypeError):
        concat(idata3, idata, dim='chain')