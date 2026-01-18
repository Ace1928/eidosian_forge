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
@pytest.mark.filterwarnings('ignore:Version 1 of dataset Australian is inactive')
@pytest.mark.parametrize('params, err_msg', [({'parser': 'pandas'}, "Sparse ARFF datasets cannot be loaded with parser='pandas'"), ({'as_frame': True}, 'Sparse ARFF datasets cannot be loaded with as_frame=True.'), ({'parser': 'pandas', 'as_frame': True}, 'Sparse ARFF datasets cannot be loaded with as_frame=True.')])
def test_fetch_openml_sparse_arff_error(monkeypatch, params, err_msg):
    """Check that we raise the expected error for sparse ARFF datasets and
    a wrong set of incompatible parameters.
    """
    pytest.importorskip('pandas')
    data_id = 292
    _monkey_patch_webbased_functions(monkeypatch, data_id, True)
    with pytest.raises(ValueError, match=err_msg):
        fetch_openml(data_id=data_id, cache=False, **params)