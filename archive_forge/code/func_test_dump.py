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
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_dump(csr_container):
    X_sparse, y_dense = _load_svmlight_local_test_file(datafile)
    X_dense = X_sparse.toarray()
    y_sparse = csr_container(np.atleast_2d(y_dense))
    X_sliced = X_sparse[np.arange(X_sparse.shape[0])]
    y_sliced = y_sparse[np.arange(y_sparse.shape[0])]
    for X in (X_sparse, X_dense, X_sliced):
        for y in (y_sparse, y_dense, y_sliced):
            for zero_based in (True, False):
                for dtype in [np.float32, np.float64, np.int32, np.int64]:
                    f = BytesIO()
                    if sp.issparse(y) and y.shape[0] == 1:
                        y = y.T
                    X_input = X.astype(dtype)
                    dump_svmlight_file(X_input, y, f, comment='test', zero_based=zero_based)
                    f.seek(0)
                    comment = f.readline()
                    comment = str(comment, 'utf-8')
                    assert 'scikit-learn %s' % sklearn.__version__ in comment
                    comment = f.readline()
                    comment = str(comment, 'utf-8')
                    assert ['one', 'zero'][zero_based] + '-based' in comment
                    X2, y2 = load_svmlight_file(f, dtype=dtype, zero_based=zero_based)
                    assert X2.dtype == dtype
                    assert_array_equal(X2.sorted_indices().indices, X2.indices)
                    X2_dense = X2.toarray()
                    if sp.issparse(X_input):
                        X_input_dense = X_input.toarray()
                    else:
                        X_input_dense = X_input
                    if dtype == np.float32:
                        assert_array_almost_equal(X_input_dense, X2_dense, 4)
                        assert_array_almost_equal(y_dense.astype(dtype, copy=False), y2, 4)
                    else:
                        assert_array_almost_equal(X_input_dense, X2_dense, 15)
                        assert_array_almost_equal(y_dense.astype(dtype, copy=False), y2, 15)