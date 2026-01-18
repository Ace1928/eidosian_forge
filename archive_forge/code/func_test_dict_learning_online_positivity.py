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
@pytest.mark.parametrize('positive_code', [False, True])
@pytest.mark.parametrize('positive_dict', [False, True])
def test_dict_learning_online_positivity(positive_code, positive_dict):
    rng = np.random.RandomState(0)
    n_components = 8
    code, dictionary = dict_learning_online(X, n_components=n_components, batch_size=4, method='cd', alpha=1, random_state=rng, positive_dict=positive_dict, positive_code=positive_code)
    if positive_dict:
        assert (dictionary >= 0).all()
    else:
        assert (dictionary < 0).any()
    if positive_code:
        assert (code >= 0).all()
    else:
        assert (code < 0).any()