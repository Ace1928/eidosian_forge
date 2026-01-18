import array
import numbers
import warnings
from collections.abc import Iterable
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from scipy import linalg
from ..preprocessing import MultiLabelBinarizer
from ..utils import check_array, check_random_state
from ..utils import shuffle as util_shuffle
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
from ..utils.random import sample_without_replacement
def sample_example():
    _, n_classes = p_w_c.shape
    y_size = n_classes + 1
    while not allow_unlabeled and y_size == 0 or y_size > n_classes:
        y_size = generator.poisson(n_labels)
    y = set()
    while len(y) != y_size:
        c = np.searchsorted(cumulative_p_c, generator.uniform(size=y_size - len(y)))
        y.update(c)
    y = list(y)
    n_words = 0
    while n_words == 0:
        n_words = generator.poisson(length)
    if len(y) == 0:
        words = generator.randint(n_features, size=n_words)
        return (words, y)
    cumulative_p_w_sample = p_w_c.take(y, axis=1).sum(axis=1).cumsum()
    cumulative_p_w_sample /= cumulative_p_w_sample[-1]
    words = np.searchsorted(cumulative_p_w_sample, generator.uniform(size=n_words))
    return (words, y)