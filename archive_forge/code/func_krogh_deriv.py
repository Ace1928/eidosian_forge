import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def krogh_deriv(x, y, axis=0):
    return KroghInterpolator(x, y, axis).derivative