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
def swap_file_mock(request, *args, **kwargs):
    url = request.get_full_url()
    if url.endswith('data/v1/download/1666876'):
        with open(corrupt_copy_path, 'rb') as f:
            corrupted_data = f.read()
        return _MockHTTPResponse(BytesIO(corrupted_data), is_gzip=True)
    else:
        return mocked_openml_url(request)