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
def test_from_ascii_string(self):
    f = b'foobar'
    self.assertEqual(b'foobar', osutils.safe_utf8(f))