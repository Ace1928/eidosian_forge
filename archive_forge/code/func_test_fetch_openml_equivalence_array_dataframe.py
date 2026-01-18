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
def test_fetch_openml_equivalence_array_dataframe(monkeypatch, parser):
    """Check the equivalence of the dataset when using `as_frame=False` and
    `as_frame=True`.
    """
    pytest.importorskip('pandas')
    data_id = 61
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response=True)
    bunch_as_frame_true = fetch_openml(data_id=data_id, as_frame=True, cache=False, parser=parser)
    bunch_as_frame_false = fetch_openml(data_id=data_id, as_frame=False, cache=False, parser=parser)
    assert_allclose(bunch_as_frame_false.data, bunch_as_frame_true.data)
    assert_array_equal(bunch_as_frame_false.target, bunch_as_frame_true.target)