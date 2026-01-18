import unittest
from unittest import mock
import pytest
from ..pkg_info import cmp_pkg_version
def test_module_removal():
    for module in _filter(MODULE_SCHEDULE):
        with pytest.raises(ImportError):
            __import__(module)
            assert False, f'Time to remove {module}'