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
def test_dict_learning_online_verbosity():
    n_components = 5
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    try:
        sys.stdout = StringIO()
        dico = MiniBatchDictionaryLearning(n_components, batch_size=4, max_iter=5, verbose=1, tol=0.1, random_state=0)
        dico.fit(X)
        dico = MiniBatchDictionaryLearning(n_components, batch_size=4, max_iter=5, verbose=1, max_no_improvement=2, random_state=0)
        dico.fit(X)
        dico = MiniBatchDictionaryLearning(n_components, batch_size=4, max_iter=5, verbose=2, random_state=0)
        dico.fit(X)
        dict_learning_online(X, n_components=n_components, batch_size=4, alpha=1, verbose=1, random_state=0)
        dict_learning_online(X, n_components=n_components, batch_size=4, alpha=1, verbose=2, random_state=0)
    finally:
        sys.stdout = old_stdout
    assert dico.components_.shape == (n_components, n_features)