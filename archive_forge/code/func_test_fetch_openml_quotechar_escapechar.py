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
def test_fetch_openml_quotechar_escapechar(monkeypatch):
    """Check that we can handle escapechar and single/double quotechar.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/25478
    """
    pd = pytest.importorskip('pandas')
    data_id = 42074
    _monkey_patch_webbased_functions(monkeypatch, data_id=data_id, gzip_response=False)
    common_params = {'as_frame': True, 'cache': False, 'data_id': data_id}
    adult_pandas = fetch_openml(parser='pandas', **common_params)
    adult_liac_arff = fetch_openml(parser='liac-arff', **common_params)
    pd.testing.assert_frame_equal(adult_pandas.frame, adult_liac_arff.frame)