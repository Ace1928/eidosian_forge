import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def bspl_antideriv(x, y, axis=0):
    return make_interp_spline(x, y, axis=axis).antiderivative()