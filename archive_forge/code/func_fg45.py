import pytest
from numpy.testing import assert_allclose, assert_equal
import numpy as np
from math import pow
from scipy import optimize
def fg45(self, x):
    return (self.f45(x), self.g45(x))