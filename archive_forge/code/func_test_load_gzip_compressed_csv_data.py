import os
import shutil
import tempfile
import warnings
from functools import partial
from importlib import resources
from pathlib import Path
from pickle import dumps, loads
import numpy as np
import pytest
from sklearn.datasets import (
from sklearn.datasets._base import (
from sklearn.datasets.tests.test_common import check_as_frame
from sklearn.preprocessing import scale
from sklearn.utils import Bunch
@pytest.mark.parametrize('filename, kwargs, expected_shape', [('diabetes_data_raw.csv.gz', {}, [442, 10]), ('diabetes_target.csv.gz', {}, [442]), ('digits.csv.gz', {'delimiter': ','}, [1797, 65])])
def test_load_gzip_compressed_csv_data(filename, kwargs, expected_shape):
    actual_data = load_gzip_compressed_csv_data(filename, **kwargs)
    assert actual_data.shape == tuple(expected_shape)