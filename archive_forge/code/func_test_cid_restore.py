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
def test_cid_restore(self):
    cb = cbook.CallbackRegistry()
    cb.connect('a', lambda: None)
    cb2 = pickle.loads(pickle.dumps(cb))
    cid = cb2.connect('c', lambda: None)
    assert cid == 1