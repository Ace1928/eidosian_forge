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
@fails_if_pypy
@pytest.mark.parametrize('data_id, parser, expected_n_categories, expected_n_floats, expected_n_ints', [(61, 'liac-arff', 1, 4, 0), (61, 'pandas', 1, 4, 0), (2, 'liac-arff', 33, 6, 0), (2, 'pandas', 33, 2, 4), (561, 'liac-arff', 1, 7, 0), (561, 'pandas', 1, 0, 7), (40589, 'liac-arff', 6, 72, 0), (40589, 'pandas', 6, 69, 3), (1119, 'liac-arff', 9, 6, 0), (1119, 'pandas', 9, 0, 6), (40966, 'liac-arff', 1, 77, 0), (40966, 'pandas', 1, 77, 0), (40945, 'liac-arff', 3, 6, 0), (40945, 'pandas', 3, 3, 3)])
@pytest.mark.parametrize('gzip_response', [True, False])
def test_fetch_openml_types_inference(monkeypatch, data_id, parser, expected_n_categories, expected_n_floats, expected_n_ints, gzip_response, datasets_column_names, datasets_missing_values):
    """Check that `fetch_openml` infer the right number of categories, integers, and
    floats."""
    pd = pytest.importorskip('pandas')
    CategoricalDtype = pd.api.types.CategoricalDtype
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response=gzip_response)
    bunch = fetch_openml(data_id=data_id, as_frame=True, cache=False, parser=parser)
    frame = bunch.frame
    n_categories = len([dtype for dtype in frame.dtypes if isinstance(dtype, CategoricalDtype)])
    n_floats = len([dtype for dtype in frame.dtypes if dtype.kind == 'f'])
    n_ints = len([dtype for dtype in frame.dtypes if dtype.kind == 'i'])
    assert n_categories == expected_n_categories
    assert n_floats == expected_n_floats
    assert n_ints == expected_n_ints
    assert frame.columns.tolist() == datasets_column_names[data_id]
    frame_feature_to_n_nan = frame.isna().sum().to_dict()
    for name, n_missing in frame_feature_to_n_nan.items():
        expected_missing = datasets_missing_values[data_id].get(name, 0)
        assert n_missing == expected_missing