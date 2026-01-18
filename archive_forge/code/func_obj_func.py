import warnings
from numbers import Integral, Real
from operator import itemgetter
import numpy as np
import scipy.optimize
from scipy.linalg import cho_solve, cholesky, solve_triangular
from ..base import BaseEstimator, MultiOutputMixin, RegressorMixin, _fit_context, clone
from ..preprocessing._data import _handle_zeros_in_scale
from ..utils import check_random_state
from ..utils._param_validation import Interval, StrOptions
from ..utils.optimize import _check_optimize_result
from .kernels import RBF, Kernel
from .kernels import ConstantKernel as C
def obj_func(theta, eval_gradient=True):
    if eval_gradient:
        lml, grad = self.log_marginal_likelihood(theta, eval_gradient=True, clone_kernel=False)
        return (-lml, -grad)
    else:
        return -self.log_marginal_likelihood(theta, clone_kernel=False)