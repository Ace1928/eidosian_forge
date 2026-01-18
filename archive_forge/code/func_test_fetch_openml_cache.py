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
def test_fetch_openml_cache(monkeypatch, gzip_response, tmpdir):

    def _mock_urlopen_raise(request, *args, **kwargs):
        raise ValueError('This mechanism intends to test correct cachehandling. As such, urlopen should never be accessed. URL: %s' % request.get_full_url())
    data_id = 61
    cache_directory = str(tmpdir.mkdir('scikit_learn_data'))
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    X_fetched, y_fetched = fetch_openml(data_id=data_id, cache=True, data_home=cache_directory, return_X_y=True, as_frame=False, parser='liac-arff')
    monkeypatch.setattr(sklearn.datasets._openml, 'urlopen', _mock_urlopen_raise)
    X_cached, y_cached = fetch_openml(data_id=data_id, cache=True, data_home=cache_directory, return_X_y=True, as_frame=False, parser='liac-arff')
    np.testing.assert_array_equal(X_fetched, X_cached)
    np.testing.assert_array_equal(y_fetched, y_cached)