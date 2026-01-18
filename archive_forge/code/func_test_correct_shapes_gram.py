import numpy as np
import pytest
from sklearn.datasets import make_sparse_coded_signal
from sklearn.linear_model import (
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
def test_correct_shapes_gram():
    assert orthogonal_mp_gram(G, Xy[:, 0], n_nonzero_coefs=5).shape == (n_features,)
    assert orthogonal_mp_gram(G, Xy, n_nonzero_coefs=5).shape == (n_features, 3)