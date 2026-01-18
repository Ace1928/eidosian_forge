import warnings
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from scipy.linalg import svd
from .base import (
from .metrics.pairwise import KERNEL_PARAMS, PAIRWISE_KERNEL_FUNCTIONS, pairwise_kernels
from .utils import check_random_state, deprecated
from .utils._param_validation import Interval, StrOptions
from .utils.extmath import safe_sparse_dot
from .utils.validation import (
@deprecated('The ``sample_interval_`` attribute was deprecated in version 1.3 and will be removed 1.5.')
@property
def sample_interval_(self):
    return self._sample_interval