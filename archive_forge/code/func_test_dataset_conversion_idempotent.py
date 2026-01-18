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
def test_dataset_conversion_idempotent(self):
    inference_data = load_arviz_data('centered_eight')
    data_set = convert_to_dataset(inference_data.posterior)
    assert isinstance(data_set, xr.Dataset)
    assert set(data_set.coords['school'].values) == {'Hotchkiss', 'Mt. Hermon', 'Choate', 'Deerfield', 'Phillips Andover', "St. Paul's", 'Lawrenceville', 'Phillips Exeter'}
    assert data_set['theta'].dims == ('chain', 'draw', 'school')