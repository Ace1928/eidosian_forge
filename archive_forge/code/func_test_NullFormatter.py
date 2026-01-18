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
def test_NullFormatter():
    formatter = mticker.NullFormatter()
    assert formatter(1.0) == ''
    assert formatter.format_data(1.0) == ''
    assert formatter.format_data_short(1.0) == ''