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
@pytest.mark.parametrize('is_latex, usetex, expected', latex_data)
def test_latex(self, is_latex, usetex, expected):
    fmt = mticker.PercentFormatter(symbol='\\{t}%', is_latex=is_latex)
    with mpl.rc_context(rc={'text.usetex': usetex}):
        assert fmt.format_pct(50, 100) == expected