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
def test_grouper():

    class Dummy:
        pass
    a, b, c, d, e = objs = [Dummy() for _ in range(5)]
    g = cbook.Grouper()
    g.join(*objs)
    assert set(list(g)[0]) == set(objs)
    assert set(g.get_siblings(a)) == set(objs)
    for other in objs[1:]:
        assert g.joined(a, other)
    g.remove(a)
    for other in objs[1:]:
        assert not g.joined(a, other)
    for A, B in itertools.product(objs[1:], objs[1:]):
        assert g.joined(A, B)