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
@pytest.mark.parametrize('parser', ['liac-arff', 'pandas'])
def test_fetch_openml_iris_pandas(monkeypatch, parser):
    """Check fetching on a numerical only dataset with string labels."""
    pd = pytest.importorskip('pandas')
    CategoricalDtype = pd.api.types.CategoricalDtype
    data_id = 61
    data_shape = (150, 4)
    target_shape = (150,)
    frame_shape = (150, 5)
    target_dtype = CategoricalDtype(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
    data_dtypes = [np.float64] * 4
    data_names = ['sepallength', 'sepalwidth', 'petallength', 'petalwidth']
    target_name = 'class'
    _monkey_patch_webbased_functions(monkeypatch, data_id, True)
    bunch = fetch_openml(data_id=data_id, as_frame=True, cache=False, parser=parser)
    data = bunch.data
    target = bunch.target
    frame = bunch.frame
    assert isinstance(data, pd.DataFrame)
    assert np.all(data.dtypes == data_dtypes)
    assert data.shape == data_shape
    assert np.all(data.columns == data_names)
    assert np.all(bunch.feature_names == data_names)
    assert bunch.target_names == [target_name]
    assert isinstance(target, pd.Series)
    assert target.dtype == target_dtype
    assert target.shape == target_shape
    assert target.name == target_name
    assert target.index.is_unique
    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == frame_shape
    assert np.all(frame.dtypes == data_dtypes + [target_dtype])
    assert frame.index.is_unique