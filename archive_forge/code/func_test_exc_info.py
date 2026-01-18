import ast
import io
import sys
import traceback
import testtools
from testtools.compat import (
from testtools.matchers import (
def test_exc_info(self):
    """After reraise exc_info matches plus some extra traceback"""
    try:
        raise ValueError('Bad value')
    except ValueError:
        _exc_info = sys.exc_info()
    try:
        reraise(*_exc_info)
    except ValueError:
        _new_exc_info = sys.exc_info()
    self.assertIs(_exc_info[0], _new_exc_info[0])
    self.assertIs(_exc_info[1], _new_exc_info[1])
    expected_tb = traceback.extract_tb(_exc_info[2])
    self.assertEqual(expected_tb, traceback.extract_tb(_new_exc_info[2])[-len(expected_tb):])