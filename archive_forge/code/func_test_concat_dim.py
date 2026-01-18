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
@pytest.mark.parametrize('dim', ['chain', 'draw'])
@pytest.mark.parametrize('copy', [True, False])
@pytest.mark.parametrize('inplace', [True, False])
@pytest.mark.parametrize('sequence', [True, False])
@pytest.mark.parametrize('reset_dim', [True, False])
def test_concat_dim(dim, copy, inplace, sequence, reset_dim):
    idata1 = from_dict(posterior={'A': np.random.randn(2, 10, 2), 'B': np.random.randn(2, 10, 5, 2)}, observed_data={'C': np.random.randn(100), 'D': np.random.randn(2, 100)})
    if inplace:
        original_idata1_id = id(idata1)
    idata2 = from_dict(posterior={'A': np.random.randn(2, 10, 2), 'B': np.random.randn(2, 10, 5, 2)}, observed_data={'C': np.random.randn(100), 'D': np.random.randn(2, 100)})
    idata3 = from_dict(posterior={'A': np.random.randn(2, 10, 2), 'B': np.random.randn(2, 10, 5, 2)}, observed_data={'C': np.random.randn(100), 'D': np.random.randn(2, 100)})
    assert concat(idata1, idata2, dim=dim, copy=copy, inplace=False, reset_dim=reset_dim) is not None
    if sequence:
        new_idata = concat((idata1, idata2, idata3), copy=copy, dim=dim, inplace=inplace, reset_dim=reset_dim)
    else:
        new_idata = concat(idata1, idata2, idata3, dim=dim, copy=copy, inplace=inplace, reset_dim=reset_dim)
    if inplace:
        assert new_idata is None
        new_idata = idata1
    assert new_idata is not None
    test_dict = {'posterior': ['A', 'B'], 'observed_data': ['C', 'D']}
    fails = check_multiple_attrs(test_dict, new_idata)
    assert not fails
    if inplace:
        assert id(new_idata) == original_idata1_id
    else:
        assert id(new_idata) != id(idata1)
    assert getattr(new_idata.posterior, dim).size == 6 if dim == 'chain' else 30
    if reset_dim:
        assert np.all(getattr(new_idata.posterior, dim).values == (np.arange(6) if dim == 'chain' else np.arange(30)))