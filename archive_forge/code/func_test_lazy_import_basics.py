import importlib
import os
import subprocess
import sys
import types
from unittest import mock
import pytest
import lazy_loader as lazy
from . import rank
from ._gaussian import gaussian
from .edges import sobel, scharr, prewitt, roberts
def test_lazy_import_basics():
    math = lazy.load('math')
    anything_not_real = lazy.load('anything_not_real')
    assert math.sin(math.pi) == pytest.approx(0, 1e-06)
    try:
        anything_not_real.pi
        raise AssertionError()
    except ModuleNotFoundError:
        pass
    assert isinstance(anything_not_real, lazy.DelayedImportErrorModule)
    try:
        anything_not_real.pi
        raise AssertionError()
    except ModuleNotFoundError:
        pass