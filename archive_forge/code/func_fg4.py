import pytest
from numpy.testing import assert_allclose, assert_equal
import numpy as np
from math import pow
from scipy import optimize
def fg4(self, x):
    return (self.f4(x), self.g4(x))