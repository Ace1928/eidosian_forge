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
@pytest.mark.parametrize('copy', [True, False])
@pytest.mark.parametrize('inplace', [True, False])
@pytest.mark.parametrize('sequence', [True, False])
def test_concat_group(copy, inplace, sequence):
    idata1 = from_dict(posterior={'A': np.random.randn(2, 10, 2), 'B': np.random.randn(2, 10, 5, 2)})
    if copy and inplace:
        original_idata1_posterior_id = id(idata1.posterior)
    idata2 = from_dict(prior={'C': np.random.randn(2, 10, 2), 'D': np.random.randn(2, 10, 5, 2)})
    idata3 = from_dict(observed_data={'E': np.random.randn(100), 'F': np.random.randn(2, 100)})
    assert concat(idata1, idata2, copy=True, inplace=False) is not None
    if sequence:
        new_idata = concat((idata1, idata2, idata3), copy=copy, inplace=inplace)
    else:
        new_idata = concat(idata1, idata2, idata3, copy=copy, inplace=inplace)
    if inplace:
        assert new_idata is None
        new_idata = idata1
    assert new_idata is not None
    test_dict = {'posterior': ['A', 'B'], 'prior': ['C', 'D'], 'observed_data': ['E', 'F']}
    fails = check_multiple_attrs(test_dict, new_idata)
    assert not fails
    if copy:
        if inplace:
            assert id(new_idata.posterior) == original_idata1_posterior_id
        else:
            assert id(new_idata.posterior) != id(idata1.posterior)
        assert id(new_idata.prior) != id(idata2.prior)
        assert id(new_idata.observed_data) != id(idata3.observed_data)
    else:
        assert id(new_idata.posterior) == id(idata1.posterior)
        assert id(new_idata.prior) == id(idata2.prior)
        assert id(new_idata.observed_data) == id(idata3.observed_data)