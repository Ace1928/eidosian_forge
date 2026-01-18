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
def test_add_groups_error(self, data_random):
    idata = data_random
    with pytest.raises(ValueError, match='One of.+must be provided.'):
        idata.add_groups()
    with pytest.raises(ValueError, match='Arguments.+xr.Dataset, xr.Dataarray or dicts'):
        idata.add_groups({'new_group': 'new_group'})
    with pytest.raises(ValueError, match='group.+already exists'):
        idata.add_groups({'posterior': idata.posterior})