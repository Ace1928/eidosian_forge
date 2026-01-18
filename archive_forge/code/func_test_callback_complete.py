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
@pytest.mark.parametrize('pickle', [True, False])
def test_callback_complete(self, pickle):
    self.is_empty()
    mini_me = Test_callback_registry()
    cid1 = self.connect(self.signal, mini_me.dummy, pickle)
    assert type(cid1) is int
    self.is_not_empty()
    cid2 = self.connect(self.signal, mini_me.dummy, pickle)
    assert cid1 == cid2
    self.is_not_empty()
    assert len(self.callbacks._func_cid_map) == 1
    assert len(self.callbacks.callbacks) == 1
    del mini_me
    self.is_empty()