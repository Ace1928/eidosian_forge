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
def test_is_inside(self):
    is_inside = osutils.is_inside
    self.assertTrue(is_inside('src', 'src/foo.c'))
    self.assertFalse(is_inside('src', 'srccontrol'))
    self.assertTrue(is_inside('src', 'src/a/a/a/foo.c'))
    self.assertTrue(is_inside('foo.c', 'foo.c'))
    self.assertFalse(is_inside('foo.c', ''))
    self.assertTrue(is_inside('', 'foo.c'))