import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def pchip_deriv_inplace(x, y, axis=0):

    class P(PchipInterpolator):

        def __call__(self, x):
            return PchipInterpolator.__call__(self, x, 1)
        pass
    return P(x, y, axis)