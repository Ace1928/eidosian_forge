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
@pytest.mark.parametrize('ntype1, ntype2', [('longdouble', 'float16'), ('float16', 'float32'), ('float32', 'double'), ('int16', 'int32'), ('int32', 'long'), ('byte', 'uint16'), ('ushort', 'uint32'), ('uint32', 'uint64'), ('uint8', 'int8')])
def test_check_pandas_sparse_invalid(ntype1, ntype2):
    """check that we raise an error with dataframe having
    sparse extension arrays with unsupported mixed dtype
    and pandas version below 1.1. pandas versions 1.1 and
    above fixed this issue so no error will be raised."""
    pd = pytest.importorskip('pandas')
    df = pd.DataFrame({'col1': pd.arrays.SparseArray([0, 1, 0], dtype=ntype1, fill_value=0), 'col2': pd.arrays.SparseArray([1, 0, 1], dtype=ntype2, fill_value=0)})
    if parse_version(pd.__version__) < parse_version('1.1'):
        err_msg = 'Pandas DataFrame with mixed sparse extension arrays'
        with pytest.raises(ValueError, match=err_msg):
            check_array(df, accept_sparse=['csr', 'csc'])
    else:
        check_array(df, accept_sparse=['csr', 'csc'])