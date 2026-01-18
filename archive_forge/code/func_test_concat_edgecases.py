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
def test_concat_edgecases(copy, inplace, sequence):
    idata = from_dict(posterior={'A': np.random.randn(2, 10, 2), 'B': np.random.randn(2, 10, 5, 2)})
    empty = concat()
    assert empty is not None
    if sequence:
        new_idata = concat([idata], copy=copy, inplace=inplace)
    else:
        new_idata = concat(idata, copy=copy, inplace=inplace)
    if inplace:
        assert new_idata is None
        new_idata = idata
    else:
        assert new_idata is not None
    test_dict = {'posterior': ['A', 'B']}
    fails = check_multiple_attrs(test_dict, new_idata)
    assert not fails
    if copy and (not inplace):
        assert id(new_idata.posterior) != id(idata.posterior)
    else:
        assert id(new_idata.posterior) == id(idata.posterior)