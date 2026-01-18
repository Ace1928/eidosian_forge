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
def test_convert_arff_data_dataframe_warning_low_memory_pandas(monkeypatch):
    """Check that we raise a warning regarding the working memory when using
    LIAC-ARFF parser."""
    pytest.importorskip('pandas')
    data_id = 1119
    _monkey_patch_webbased_functions(monkeypatch, data_id, True)
    msg = 'Could not adhere to working_memory config.'
    with pytest.warns(UserWarning, match=msg):
        with config_context(working_memory=1e-06):
            fetch_openml(data_id=data_id, as_frame=True, cache=False, parser='liac-arff')