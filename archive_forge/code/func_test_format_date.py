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
def test_format_date(self):
    self.assertRaises(osutils.UnsupportedTimezoneFormat, osutils.format_date, 0, timezone='foo')
    self.assertIsInstance(osutils.format_date(0), str)
    self.assertIsInstance(osutils.format_local_date(0), str)