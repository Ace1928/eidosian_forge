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
def test_load_csv_data_with_descr():
    data_file_name = 'iris.csv'
    descr_file_name = 'iris.rst'
    res_without_descr = load_csv_data(data_file_name=data_file_name)
    res_with_descr = load_csv_data(data_file_name=data_file_name, descr_file_name=descr_file_name)
    assert len(res_with_descr) == 4
    assert len(res_without_descr) == 3
    np.testing.assert_array_equal(res_with_descr[0], res_without_descr[0])
    np.testing.assert_array_equal(res_with_descr[1], res_without_descr[1])
    np.testing.assert_array_equal(res_with_descr[2], res_without_descr[2])
    assert res_with_descr[-1].startswith('.. _iris_dataset:')