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
def test_dataset_with_openml_warning(monkeypatch, gzip_response):
    data_id = 3
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    msg = 'OpenML raised a warning on the dataset. It might be unusable. Warning:'
    with pytest.warns(UserWarning, match=msg):
        fetch_openml(data_id=data_id, cache=False, as_frame=False, parser='liac-arff')