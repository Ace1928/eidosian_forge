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
@pytest.mark.parametrize('positive_dict', [False, True])
def test_minibatch_dictionary_learning_lars(positive_dict):
    n_components = 8
    dico = MiniBatchDictionaryLearning(n_components, batch_size=4, max_iter=10, transform_algorithm='lars', random_state=0, positive_dict=positive_dict, fit_algorithm='cd').fit(X)
    if positive_dict:
        assert (dico.components_ >= 0).all()
    else:
        assert (dico.components_ < 0).any()