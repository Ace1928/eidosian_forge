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
def test_dict_learning_online_initialization():
    n_components = 12
    rng = np.random.RandomState(0)
    V = rng.randn(n_components, n_features)
    dico = MiniBatchDictionaryLearning(n_components, batch_size=4, max_iter=0, dict_init=V, random_state=0).fit(X)
    assert_array_equal(dico.components_, V)