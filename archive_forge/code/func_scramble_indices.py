import numpy as np
import pytest
from scipy import sparse
from sklearn import base, datasets, linear_model, svm
from sklearn.datasets import load_digits, make_blobs, make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm.tests import test_svm
from sklearn.utils._testing import (
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.fixes import (
def scramble_indices(X):
    new_data = []
    new_indices = []
    for i in range(1, len(X.indptr)):
        row_slice = slice(*X.indptr[i - 1:i + 1])
        new_data.extend(X.data[row_slice][::-1])
        new_indices.extend(X.indices[row_slice][::-1])
    return csr_container((new_data, new_indices, X.indptr), shape=X.shape)