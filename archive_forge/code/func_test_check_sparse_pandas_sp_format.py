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
@pytest.mark.parametrize('sp_format', [True, 'csr', 'csc', 'coo', 'bsr'])
def test_check_sparse_pandas_sp_format(sp_format):
    pd = pytest.importorskip('pandas')
    sp_mat = _sparse_random_matrix(10, 3)
    sdf = pd.DataFrame.sparse.from_spmatrix(sp_mat)
    result = check_array(sdf, accept_sparse=sp_format)
    if sp_format is True:
        sp_format = 'coo'
    assert sp.issparse(result)
    assert result.format == sp_format
    assert_allclose_dense_sparse(sp_mat, result)