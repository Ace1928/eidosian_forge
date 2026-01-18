import pkgutil
import types
import importlib
import warnings
from importlib import import_module
import pytest
import scipy
def test_dir_testing():
    """Assert that output of dir has only one "testing/tester"
    attribute without duplicate"""
    assert len(dir(scipy)) == len(set(dir(scipy)))