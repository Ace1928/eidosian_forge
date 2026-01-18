import gzip
import json
import os
import re
from functools import partial
from importlib import resources
from io import BytesIO
from urllib.error import HTTPError
import numpy as np
import pytest
import scipy.sparse
import sklearn
from sklearn import config_context
from sklearn.datasets import fetch_openml as fetch_openml_orig
from sklearn.datasets._openml import (
from sklearn.utils import Bunch, check_pandas_support
from sklearn.utils._testing import (
@pytest.mark.parametrize('gzip_response', [True, False])
def test_fetch_openml_no_target(monkeypatch, gzip_response):
    """Check that we can get a dataset without target."""
    data_id = 61
    target_column = None
    expected_observations = 150
    expected_features = 5
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    data = fetch_openml(data_id=data_id, target_column=target_column, cache=False, as_frame=False, parser='liac-arff')
    assert data.data.shape == (expected_observations, expected_features)
    assert data.target is None