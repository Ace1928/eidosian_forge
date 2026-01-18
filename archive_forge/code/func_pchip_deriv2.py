import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def pchip_deriv2(x, y, axis=0):
    return pchip(x, y, axis).derivative(2)