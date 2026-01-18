import gzip
import os
import shutil
from bz2 import BZ2File
from importlib import resources
from io import BytesIO
from tempfile import NamedTemporaryFile
import numpy as np
import pytest
import scipy.sparse as sp
import sklearn
from sklearn.datasets import dump_svmlight_file, load_svmlight_file, load_svmlight_files
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_load_invalid_file2():
    with pytest.raises(ValueError):
        data_path = _svmlight_local_test_file_path(datafile)
        invalid_path = _svmlight_local_test_file_path(invalidfile)
        load_svmlight_files([str(data_path), str(invalid_path), str(data_path)])