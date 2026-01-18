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
def test_load_svmlight_file_multilabel():
    X, y = _load_svmlight_local_test_file(multifile, multilabel=True)
    assert y == [(0, 1), (2,), (), (1, 2)]