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
def test_fetch_openml_iris_warn_multiple_version(monkeypatch, gzip_response):
    """Check that a warning is raised when multiple versions exist and no version is
    requested."""
    data_id = 61
    data_name = 'iris'
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    msg = re.escape('Multiple active versions of the dataset matching the name iris exist. Versions may be fundamentally different, returning version 1. Available versions:\n- version 1, status: active\n  url: https://www.openml.org/search?type=data&id=61\n- version 3, status: active\n  url: https://www.openml.org/search?type=data&id=969\n')
    with pytest.warns(UserWarning, match=msg):
        fetch_openml(name=data_name, as_frame=False, cache=False, parser='liac-arff')