from __future__ import annotations
import itertools
import pickle
from typing import Any
from unittest.mock import patch, Mock
from datetime import datetime, date, timedelta
import numpy as np
from numpy.testing import (assert_array_equal, assert_approx_equal,
import pytest
from matplotlib import _api, cbook
import matplotlib.colors as mcolors
from matplotlib.cbook import delete_masked_points, strip_math
def test_format_approx():
    f = cbook._format_approx
    assert f(0, 1) == '0'
    assert f(0, 2) == '0'
    assert f(0, 3) == '0'
    assert f(-0.0123, 1) == '-0'
    assert f(1e-07, 5) == '0'
    assert f(0.0012345600001, 5) == '0.00123'
    assert f(-0.0012345600001, 5) == '-0.00123'
    assert f(0.0012345600001, 8) == f(0.0012345600001, 10) == '0.00123456'