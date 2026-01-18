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
def convert_numerical_and_categorical_dtypes(series):
    pandas_series = frame_pandas[series.name]
    if pd.api.types.is_numeric_dtype(pandas_series):
        return series.astype(pandas_series.dtype)
    elif isinstance(pandas_series.dtype, pd.CategoricalDtype):
        return series.cat.rename_categories(pandas_series.cat.categories)
    else:
        return series