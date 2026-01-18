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
def test_failure_to_load(self):
    self._try_loading()
    self.assertLength(1, osutils._extension_load_failures)
    self.assertEqual(osutils._extension_load_failures[0], "No module named 'breezy._fictional_extension_py'")