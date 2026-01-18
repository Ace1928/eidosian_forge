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
def test_callbackregistry_blocking():

    def raise_handler(excp):
        raise excp
    cb = cbook.CallbackRegistry(exception_handler=raise_handler)

    def test_func1():
        raise ValueError('1 should be blocked')

    def test_func2():
        raise ValueError('2 should be blocked')
    cb.connect('test1', test_func1)
    cb.connect('test2', test_func2)
    with cb.blocked():
        cb.process('test1')
        cb.process('test2')
    with cb.blocked(signal='test1'):
        cb.process('test1')
        with pytest.raises(ValueError, match='2 should be blocked'):
            cb.process('test2')
    with pytest.raises(ValueError, match='1 should be blocked'):
        cb.process('test1')
    with pytest.raises(ValueError, match='2 should be blocked'):
        cb.process('test2')