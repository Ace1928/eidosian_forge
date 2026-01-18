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
def test_load_gzip_compressed_csv_data_with_descr():
    data_file_name = 'diabetes_target.csv.gz'
    descr_file_name = 'diabetes.rst'
    expected_data = load_gzip_compressed_csv_data(data_file_name=data_file_name)
    actual_data, descr = load_gzip_compressed_csv_data(data_file_name=data_file_name, descr_file_name=descr_file_name)
    np.testing.assert_array_equal(actual_data, expected_data)
    assert descr.startswith('.. _diabetes_dataset:')