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
@pytest.mark.parametrize('data_id, params, err_type, err_msg', [(40675, {'name': 'glass2'}, ValueError, 'No active dataset glass2 found'), (61, {'data_id': 61, 'target_column': ['sepalwidth', 'class']}, ValueError, 'Can only handle homogeneous multi-target datasets'), (40945, {'data_id': 40945, 'as_frame': False}, ValueError, 'STRING attributes are not supported for array representation. Try as_frame=True'), (2, {'data_id': 2, 'target_column': 'family', 'as_frame': True}, ValueError, "Target column 'family'"), (2, {'data_id': 2, 'target_column': 'family', 'as_frame': False}, ValueError, "Target column 'family'"), (61, {'data_id': 61, 'target_column': 'undefined'}, KeyError, "Could not find target_column='undefined'"), (61, {'data_id': 61, 'target_column': ['undefined', 'class']}, KeyError, "Could not find target_column='undefined'")])
@pytest.mark.parametrize('parser', ['liac-arff', 'pandas'])
def test_fetch_openml_error(monkeypatch, gzip_response, data_id, params, err_type, err_msg, parser):
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    if params.get('as_frame', True) or parser == 'pandas':
        pytest.importorskip('pandas')
    with pytest.raises(err_type, match=err_msg):
        fetch_openml(cache=False, parser=parser, **params)