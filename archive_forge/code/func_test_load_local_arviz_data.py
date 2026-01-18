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
def test_load_local_arviz_data():
    inference_data = load_arviz_data('centered_eight')
    assert isinstance(inference_data, InferenceData)
    assert set(inference_data.observed_data.obs.coords['school'].values) == {'Hotchkiss', 'Mt. Hermon', 'Choate', 'Deerfield', 'Phillips Andover', "St. Paul's", 'Lawrenceville', 'Phillips Exeter'}
    assert inference_data.posterior['theta'].dims == ('chain', 'draw', 'school')