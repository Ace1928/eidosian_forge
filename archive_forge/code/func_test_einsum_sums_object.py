import itertools
import sys
import platform
import pytest
import numpy as np
from numpy.testing import (
def test_einsum_sums_object(self):
    self.check_einsum_sums('object')
    self.check_einsum_sums('object', True)