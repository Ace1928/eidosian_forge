import warnings
from math import log, sqrt
from numbers import Integral, Real
import numpy as np
from scipy import linalg
from ..base import (
from ..exceptions import ConvergenceWarning
from ..utils import check_random_state
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import fast_logdet, randomized_svd, squared_norm
from ..utils.validation import check_is_fitted
def my_svd(X):
    _, s, Vt = randomized_svd(X, n_components, random_state=random_state, n_iter=self.iterated_power)
    return (s, Vt, squared_norm(X) - squared_norm(s))