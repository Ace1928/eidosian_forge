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
def test_add_groups(self, data_random):
    data = np.random.normal(size=(4, 500, 8))
    idata = data_random
    idata.add_groups({'prior': {'a': data[..., 0], 'b': data}})
    assert 'prior' in idata._groups
    assert isinstance(idata.prior, xr.Dataset)
    assert hasattr(idata, 'prior')
    idata.add_groups(warmup_posterior={'a': data[..., 0], 'b': data})
    assert 'warmup_posterior' in idata._groups_all
    assert isinstance(idata.warmup_posterior, xr.Dataset)
    assert hasattr(idata, 'warmup_posterior')