import itertools
import sys
import platform
import pytest
import numpy as np
from numpy.testing import (
def test_hadamard_like_products(self):
    self.optimize_compare('a,ab,abc->abc')
    self.optimize_compare('a,b,ab->ab')