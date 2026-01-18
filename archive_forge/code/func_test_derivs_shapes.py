import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def test_derivs_shapes():
    for ip in [KroghInterpolator, BarycentricInterpolator]:

        def interpolator_derivs(x, y, axis=0):
            return ip(x, y, axis).derivatives
        for s1 in SHAPES:
            for s2 in SHAPES:
                for axis in range(-len(s2), len(s2)):
                    check_shape(interpolator_derivs, s1, s2, (6,), axis)