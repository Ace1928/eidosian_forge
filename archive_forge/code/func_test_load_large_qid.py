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
@pytest.mark.skip('testing the overflow of 32 bit sparse indexing requires a large amount of memory')
def test_load_large_qid():
    """
    load large libsvm / svmlight file with qid attribute. Tests 64-bit query ID
    """
    data = b'\n'.join(('3 qid:{0} 1:0.53 2:0.12\n2 qid:{0} 1:0.13 2:0.1'.format(i).encode() for i in range(1, 40 * 1000 * 1000)))
    X, y, qid = load_svmlight_file(BytesIO(data), query_id=True)
    assert_array_equal(y[-4:], [3, 2, 3, 2])
    assert_array_equal(np.unique(qid), np.arange(1, 40 * 1000 * 1000))