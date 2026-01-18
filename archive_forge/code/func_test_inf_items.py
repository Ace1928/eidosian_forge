import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_inf_items(self):
    self._assert_func(np.inf, np.inf)
    self._assert_func([np.inf], [np.inf])
    self._test_not_equal(np.inf, [np.inf])