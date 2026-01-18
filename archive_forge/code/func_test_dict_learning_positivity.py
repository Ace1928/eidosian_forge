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
@pytest.mark.parametrize('transform_algorithm', ['lasso_lars', 'lasso_cd', 'threshold'])
@pytest.mark.parametrize('positive_code', [False, True])
@pytest.mark.parametrize('positive_dict', [False, True])
def test_dict_learning_positivity(transform_algorithm, positive_code, positive_dict):
    n_components = 5
    dico = DictionaryLearning(n_components, transform_algorithm=transform_algorithm, random_state=0, positive_code=positive_code, positive_dict=positive_dict, fit_algorithm='cd').fit(X)
    code = dico.transform(X)
    if positive_dict:
        assert (dico.components_ >= 0).all()
    else:
        assert (dico.components_ < 0).any()
    if positive_code:
        assert (code >= 0).all()
    else:
        assert (code < 0).any()