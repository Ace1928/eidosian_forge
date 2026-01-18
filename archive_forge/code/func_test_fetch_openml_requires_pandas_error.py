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
@pytest.mark.parametrize('params', [{'as_frame': True, 'parser': 'auto'}, {'as_frame': 'auto', 'parser': 'auto'}, {'as_frame': False, 'parser': 'pandas'}, {'as_frame': False, 'parser': 'auto'}])
def test_fetch_openml_requires_pandas_error(monkeypatch, params):
    """Check that we raise the proper errors when we require pandas."""
    data_id = 1119
    try:
        check_pandas_support('test_fetch_openml_requires_pandas')
    except ImportError:
        _monkey_patch_webbased_functions(monkeypatch, data_id, True)
        err_msg = 'requires pandas to be installed. Alternatively, explicitly'
        with pytest.raises(ImportError, match=err_msg):
            fetch_openml(data_id=data_id, **params)
    else:
        raise SkipTest('This test requires pandas to not be installed.')