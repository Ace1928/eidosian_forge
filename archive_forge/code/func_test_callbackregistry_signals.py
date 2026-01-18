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
def test_callbackregistry_signals():
    cr = cbook.CallbackRegistry(signals=['foo'])
    results = []

    def cb(x):
        results.append(x)
    cr.connect('foo', cb)
    with pytest.raises(ValueError):
        cr.connect('bar', cb)
    cr.process('foo', 1)
    with pytest.raises(ValueError):
        cr.process('bar', 1)
    assert results == [1]