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
def test_dump_read_only(tmp_path):
    """Ensure that there is no ValueError when dumping a read-only `X`.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/28026
    """
    rng = np.random.RandomState(42)
    X = rng.randn(5, 2)
    y = rng.randn(5)
    X, y = create_memmap_backed_data([X, y])
    save_path = str(tmp_path / 'svm_read_only')
    dump_svmlight_file(X, y, save_path)