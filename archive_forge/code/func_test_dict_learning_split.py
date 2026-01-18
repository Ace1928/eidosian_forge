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
def test_dict_learning_split():
    n_components = 5
    dico = DictionaryLearning(n_components, transform_algorithm='threshold', random_state=0)
    code = dico.fit(X).transform(X)
    dico.split_sign = True
    split_code = dico.transform(X)
    assert_array_almost_equal(split_code[:, :n_components] - split_code[:, n_components:], code)