from inspect import signature
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.gaussian_process.kernels import (
from sklearn.metrics.pairwise import (
from sklearn.utils._testing import (
@pytest.mark.parametrize('kernel', kernels)
def test_kernel_clone_after_set_params(kernel):
    bounds = (1e-05, 100000.0)
    kernel_cloned = clone(kernel)
    params = kernel.get_params()
    isotropic_kernels = (ExpSineSquared, RationalQuadratic)
    if 'length_scale' in params and (not isinstance(kernel, isotropic_kernels)):
        length_scale = params['length_scale']
        if np.iterable(length_scale):
            params['length_scale'] = length_scale[0]
            params['length_scale_bounds'] = bounds
        else:
            params['length_scale'] = [length_scale] * 2
            params['length_scale_bounds'] = bounds * 2
        kernel_cloned.set_params(**params)
        kernel_cloned_clone = clone(kernel_cloned)
        assert kernel_cloned_clone.get_params() == kernel_cloned.get_params()
        assert id(kernel_cloned_clone) != id(kernel_cloned)
        check_hyperparameters_equal(kernel_cloned, kernel_cloned_clone)