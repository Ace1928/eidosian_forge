import itertools
import warnings
from functools import partial
import numpy as np
import pytest
import sklearn
from sklearn.base import clone
from sklearn.decomposition import (
from sklearn.decomposition._dict_learning import _update_dict
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_array
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
from sklearn.utils.parallel import Parallel
def test_dict_learning_lassocd_readonly_data():
    n_components = 12
    with TempMemmap(X) as X_read_only:
        dico = DictionaryLearning(n_components, transform_algorithm='lasso_cd', transform_alpha=0.001, random_state=0, n_jobs=4)
        with ignore_warnings(category=ConvergenceWarning):
            code = dico.fit(X_read_only).transform(X_read_only)
        assert_array_almost_equal(np.dot(code, dico.components_), X_read_only, decimal=2)