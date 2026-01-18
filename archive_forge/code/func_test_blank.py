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
def test_blank(self):
    formatter = mticker.LogFormatterExponent(base=10, labelOnlyBase=True)
    formatter.create_dummy_axis()
    formatter.axis.set_view_interval(1, 10)
    assert formatter(10 ** 0.1) == ''