import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def akima_deriv(x, y, axis=0):
    return Akima1DInterpolator(x, y, axis).derivative()