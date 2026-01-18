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
def test_callbackregistry_default_exception_handler(capsys, monkeypatch):
    cb = cbook.CallbackRegistry()
    cb.connect('foo', lambda: None)
    monkeypatch.setattr(cbook, '_get_running_interactive_framework', lambda: None)
    with pytest.raises(TypeError):
        cb.process('foo', 'argument mismatch')
    outerr = capsys.readouterr()
    assert outerr.out == outerr.err == ''
    monkeypatch.setattr(cbook, '_get_running_interactive_framework', lambda: 'not-none')
    cb.process('foo', 'argument mismatch')
    outerr = capsys.readouterr()
    assert outerr.out == ''
    assert 'takes 0 positional arguments but 1 was given' in outerr.err