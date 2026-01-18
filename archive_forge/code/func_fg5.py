import pytest
from numpy.testing import assert_allclose, assert_equal
import numpy as np
from math import pow
from scipy import optimize
def fg5(self, x):
    return (self.f5(x), self.g5(x))