from contextlib import nullcontext
import itertools
import locale
import logging
import re
from packaging.version import parse as parse_version
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
def test_basic_with_offset(self):
    loc = mticker.MultipleLocator(base=3.147, offset=1.2)
    test_value = np.array([-8.241, -5.094, -1.947, 1.2, 4.347, 7.494, 10.641])
    assert_almost_equal(loc.tick_values(-7, 10), test_value)