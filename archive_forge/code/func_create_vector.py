import pytest
import numpy as np
from numpy import arange, add, array, eye, copy, sqrt
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.special import comb
from scipy.linalg import (toeplitz, hankel, circulant, hadamard, leslie, dft,
from numpy.linalg import cond
def create_vector(self, n, cpx):
    """Make a complex or real test vector of length n."""
    x = np.linspace(-2.5, 2.2, n)
    if cpx:
        x = x + 1j * np.linspace(-1.5, 3.1, n)
    return x