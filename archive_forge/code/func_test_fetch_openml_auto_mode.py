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
@pytest.mark.filterwarnings('ignore:Version 1 of dataset Australian is inactive')
@pytest.mark.parametrize('data_id, data_type', [(61, 'dataframe'), (292, 'sparse')])
def test_fetch_openml_auto_mode(monkeypatch, data_id, data_type):
    """Check the auto mode of `fetch_openml`."""
    pd = pytest.importorskip('pandas')
    _monkey_patch_webbased_functions(monkeypatch, data_id, True)
    data = fetch_openml(data_id=data_id, as_frame='auto', cache=False)
    klass = pd.DataFrame if data_type == 'dataframe' else scipy.sparse.csr_matrix
    assert isinstance(data.data, klass)