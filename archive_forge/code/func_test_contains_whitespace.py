import errno
import os
import select
import socket
import sys
import tempfile
import time
from io import BytesIO
from .. import errors, osutils, tests, trace, win32utils
from . import features, file_utils, test__walkdirs_win32
from .scenarios import load_tests_apply_scenarios
def test_contains_whitespace(self):
    self.assertTrue(osutils.contains_whitespace(' '))
    self.assertTrue(osutils.contains_whitespace('hello there'))
    self.assertTrue(osutils.contains_whitespace('hellothere\n'))
    self.assertTrue(osutils.contains_whitespace('hello\nthere'))
    self.assertTrue(osutils.contains_whitespace('hello\rthere'))
    self.assertTrue(osutils.contains_whitespace('hello\tthere'))
    self.assertFalse(osutils.contains_whitespace(''))
    self.assertFalse(osutils.contains_whitespace('hellothere'))
    self.assertFalse(osutils.contains_whitespace('hello\xa0there'))