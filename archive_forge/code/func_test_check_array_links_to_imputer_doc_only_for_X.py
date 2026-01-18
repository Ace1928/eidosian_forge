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
@pytest.mark.parametrize('input_name', ['X', 'y', 'sample_weight'])
@pytest.mark.parametrize('retype', [np.asarray, sp.csr_matrix])
def test_check_array_links_to_imputer_doc_only_for_X(input_name, retype):
    data = retype(np.arange(4).reshape(2, 2).astype(np.float64))
    data[0, 0] = np.nan
    estimator = SVR()
    extended_msg = f'\n{estimator.__class__.__name__} does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values'
    with pytest.raises(ValueError, match=f'Input {input_name} contains NaN') as ctx:
        check_array(data, estimator=estimator, input_name=input_name, accept_sparse=True)
    if input_name == 'X':
        assert extended_msg in ctx.value.args[0]
    else:
        assert extended_msg not in ctx.value.args[0]
    if input_name == 'X':
        with pytest.raises(ValueError, match=f'Input {input_name} contains NaN') as ctx:
            SVR().fit(data, np.ones(data.shape[0]))
        assert extended_msg in ctx.value.args[0]