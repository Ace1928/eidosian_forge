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
def test_fancy_rename(self):
    self.create_file('a', b'something in a\n')
    self._fancy_rename('a', 'b')
    self.assertPathDoesNotExist('a')
    self.assertPathExists('b')
    self.check_file_contents('b', b'something in a\n')
    self.create_file('a', b'new something in a\n')
    self._fancy_rename('b', 'a')
    self.check_file_contents('a', b'something in a\n')