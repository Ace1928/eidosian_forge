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
def test_fetch_openml_overwrite_default_params_read_csv(monkeypatch):
    """Check that we can overwrite the default parameters of `read_csv`."""
    pytest.importorskip('pandas')
    data_id = 1590
    _monkey_patch_webbased_functions(monkeypatch, data_id=data_id, gzip_response=False)
    common_params = {'data_id': data_id, 'as_frame': True, 'cache': False, 'parser': 'pandas'}
    adult_without_spaces = fetch_openml(**common_params)
    adult_with_spaces = fetch_openml(**common_params, read_csv_kwargs={'skipinitialspace': False})
    assert all((cat.startswith(' ') for cat in adult_with_spaces.frame['class'].cat.categories))
    assert not any((cat.startswith(' ') for cat in adult_without_spaces.frame['class'].cat.categories))