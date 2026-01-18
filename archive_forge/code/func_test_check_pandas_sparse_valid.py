import numbers
import re
import warnings
from itertools import product
from operator import itemgetter
from tempfile import NamedTemporaryFile
import numpy as np
import pytest
import scipy.sparse as sp
from pytest import importorskip
import sklearn
from sklearn._config import config_context
from sklearn._min_dependencies import dependent_packages
from sklearn.base import BaseEstimator
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError, PositiveSpectrumWarning
from sklearn.linear_model import ARDRegression
from sklearn.metrics.tests.test_score_objects import EstimatorWithFit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.random_projection import _sparse_random_matrix
from sklearn.svm import SVR
from sklearn.utils import (
from sklearn.utils._mocking import (
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import _NotAnArray
from sklearn.utils.fixes import (
from sklearn.utils.validation import (
@pytest.mark.parametrize('ntype1, ntype2, expected_subtype', [('double', 'longdouble', np.floating), ('float16', 'half', np.floating), ('single', 'float32', np.floating), ('double', 'float64', np.floating), ('int8', 'byte', np.integer), ('short', 'int16', np.integer), ('intc', 'int32', np.integer), ('intp', 'long', np.integer), ('int', 'long', np.integer), ('int64', 'longlong', np.integer), ('int_', 'intp', np.integer), ('ubyte', 'uint8', np.unsignedinteger), ('uint16', 'ushort', np.unsignedinteger), ('uintc', 'uint32', np.unsignedinteger), ('uint', 'uint64', np.unsignedinteger), ('uintp', 'ulonglong', np.unsignedinteger)])
def test_check_pandas_sparse_valid(ntype1, ntype2, expected_subtype):
    pd = pytest.importorskip('pandas')
    df = pd.DataFrame({'col1': pd.arrays.SparseArray([0, 1, 0], dtype=ntype1, fill_value=0), 'col2': pd.arrays.SparseArray([1, 0, 1], dtype=ntype2, fill_value=0)})
    arr = check_array(df, accept_sparse=['csr', 'csc'])
    assert np.issubdtype(arr.dtype, expected_subtype)