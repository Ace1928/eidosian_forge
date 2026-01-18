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
def test_dict_learning_online_estimator_shapes():
    n_components = 5
    dico = MiniBatchDictionaryLearning(n_components, batch_size=4, max_iter=5, random_state=0)
    dico.fit(X)
    assert dico.components_.shape == (n_components, n_features)