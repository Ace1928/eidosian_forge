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
def test_inference_concat_keeps_all_fields():
    """From failures observed in issue #907"""
    idata1 = from_dict(posterior={'A': [1, 2, 3, 4]}, sample_stats={'B': [2, 3, 4, 5]})
    idata2 = from_dict(prior={'C': [1, 2, 3, 4]}, observed_data={'D': [2, 3, 4, 5]})
    idata_c1 = concat(idata1, idata2)
    idata_c2 = concat(idata2, idata1)
    test_dict = {'posterior': ['A'], 'sample_stats': ['B'], 'prior': ['C'], 'observed_data': ['D']}
    fails_c1 = check_multiple_attrs(test_dict, idata_c1)
    assert not fails_c1
    fails_c2 = check_multiple_attrs(test_dict, idata_c2)
    assert not fails_c2