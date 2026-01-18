import pickle
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pytest import approx
from scipy.optimize import (
from scipy.special import logsumexp
from sklearn._loss.link import IdentityLink, _inclusive_low_high
from sklearn._loss.loss import (
from sklearn.utils import _IS_WASM, assert_all_finite
from sklearn.utils._testing import create_memmap_backed_data, skip_if_32bit
def numerical_derivative(func, x, eps):
    """Helper function for numerical (first) derivatives."""
    h = np.full_like(x, fill_value=eps)
    f_minus_2h = func(x - 2 * h)
    f_minus_1h = func(x - h)
    f_plus_1h = func(x + h)
    f_plus_2h = func(x + 2 * h)
    return (-f_plus_2h + 8 * f_plus_1h - 8 * f_minus_1h + f_minus_2h) / (12.0 * eps)