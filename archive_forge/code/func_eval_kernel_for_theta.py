from inspect import signature
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.gaussian_process.kernels import (
from sklearn.metrics.pairwise import (
from sklearn.utils._testing import (
def eval_kernel_for_theta(theta):
    kernel_clone = kernel.clone_with_theta(theta)
    K = kernel_clone(X, eval_gradient=False)
    return K