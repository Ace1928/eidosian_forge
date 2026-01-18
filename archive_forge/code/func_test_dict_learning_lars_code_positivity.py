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
def test_dict_learning_lars_code_positivity():
    n_components = 5
    dico = DictionaryLearning(n_components, transform_algorithm='lars', random_state=0, positive_code=True, fit_algorithm='cd').fit(X)
    err_msg = "Positive constraint not supported for '{}' coding method."
    err_msg = err_msg.format('lars')
    with pytest.raises(ValueError, match=err_msg):
        dico.transform(X)