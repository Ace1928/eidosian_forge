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
@pytest.mark.parametrize('parser', ('liac-arff', 'pandas'))
def test_fetch_openml_with_ignored_feature(monkeypatch, gzip_response, parser):
    """Check that we can load the "zoo" dataset.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/14340
    """
    if parser == 'pandas':
        pytest.importorskip('pandas')
    data_id = 62
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    dataset = sklearn.datasets.fetch_openml(data_id=data_id, cache=False, as_frame=False, parser=parser)
    assert dataset is not None
    assert dataset['data'].shape == (101, 16)
    assert 'animal' not in dataset['feature_names']