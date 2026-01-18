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
def test_attach_same_module_and_attr_name():
    from lazy_loader.tests import fake_pkg
    assert isinstance(fake_pkg.some_func, types.FunctionType)
    assert isinstance(fake_pkg.some_func, types.FunctionType)
    from lazy_loader.tests.fake_pkg.some_func import some_func
    assert isinstance(some_func, types.FunctionType)