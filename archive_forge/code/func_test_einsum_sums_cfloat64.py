import itertools
import sys
import platform
import pytest
import numpy as np
from numpy.testing import (
def test_einsum_sums_cfloat64(self):
    self.check_einsum_sums('c8')
    self.check_einsum_sums('c8', True)