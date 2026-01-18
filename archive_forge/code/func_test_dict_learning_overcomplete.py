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
def test_dict_learning_overcomplete():
    n_components = 12
    dico = DictionaryLearning(n_components, random_state=0).fit(X)
    assert dico.components_.shape == (n_components, n_features)