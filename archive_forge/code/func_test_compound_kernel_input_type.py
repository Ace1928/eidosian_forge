from inspect import signature
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.gaussian_process.kernels import (
from sklearn.metrics.pairwise import (
from sklearn.utils._testing import (
def test_compound_kernel_input_type():
    kernel = CompoundKernel([WhiteKernel(noise_level=3.0)])
    assert not kernel.requires_vector_input
    kernel = CompoundKernel([WhiteKernel(noise_level=3.0), RBF(length_scale=2.0)])
    assert kernel.requires_vector_input