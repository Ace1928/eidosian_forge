import itertools
import os
import warnings
from functools import partial
import numpy as np
import pytest
from numpy.testing import (
from scipy import sparse
from sklearn import config_context
from sklearn.base import clone
from sklearn.datasets import load_iris, make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model._logistic import (
from sklearn.linear_model._logistic import (
from sklearn.linear_model._logistic import (
from sklearn.metrics import get_scorer, log_loss
from sklearn.model_selection import (
from sklearn.preprocessing import LabelEncoder, StandardScaler, scale
from sklearn.svm import l1_min_c
from sklearn.utils import _IS_32BIT, compute_class_weight, shuffle
from sklearn.utils._testing import ignore_warnings, skip_if_no_parallel
from sklearn.utils.fixes import COO_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('max_iter', np.arange(1, 5))
@pytest.mark.parametrize('multi_class', ['ovr', 'multinomial'])
@pytest.mark.parametrize('solver, message', [('newton-cg', 'newton-cg failed to converge. Increase the number of iterations.'), ('liblinear', 'Liblinear failed to converge, increase the number of iterations.'), ('sag', 'The max_iter was reached which means the coef_ did not converge'), ('saga', 'The max_iter was reached which means the coef_ did not converge'), ('lbfgs', 'lbfgs failed to converge'), ('newton-cholesky', 'Newton solver did not converge after [0-9]* iterations')])
def test_max_iter(max_iter, multi_class, solver, message):
    X, y_bin = (iris.data, iris.target.copy())
    y_bin[y_bin == 2] = 0
    if solver in ('liblinear', 'newton-cholesky') and multi_class == 'multinomial':
        pytest.skip("'multinomial' is not supported by liblinear and newton-cholesky")
    if solver == 'newton-cholesky' and max_iter > 1:
        pytest.skip('solver newton-cholesky might converge very fast')
    lr = LogisticRegression(max_iter=max_iter, tol=1e-15, multi_class=multi_class, random_state=0, solver=solver)
    with pytest.warns(ConvergenceWarning, match=message):
        lr.fit(X, y_bin)
    assert lr.n_iter_[0] == max_iter