import itertools
import sys
import platform
import pytest
import numpy as np
from numpy.testing import (
def test_einsum_sums_float16(self):
    self.check_einsum_sums('f2')