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
@pytest.mark.parametrize('inplace', [True, False])
def test_extend_xr_method(self, data_random, inplace):
    idata = data_random
    idata_copy = deepcopy(idata)
    kwargs = {'groups': 'posterior_groups'}
    if inplace:
        idata_copy.sum(dim='draw', inplace=inplace, **kwargs)
    else:
        idata2 = idata_copy.sum(dim='draw', inplace=inplace, **kwargs)
        assert idata2 is not idata_copy
        idata_copy = idata2
    assert_identical(idata_copy.posterior, idata.posterior.sum(dim='draw'))
    assert_identical(idata_copy.posterior_predictive, idata.posterior_predictive.sum(dim='draw'))
    assert_identical(idata_copy.observed_data, idata.observed_data)