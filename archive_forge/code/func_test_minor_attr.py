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
def test_minor_attr(self):
    loc = mticker.LogitLocator(nbins=100)
    assert not loc.minor
    loc.minor = True
    assert loc.minor
    loc.set_params(minor=False)
    assert not loc.minor