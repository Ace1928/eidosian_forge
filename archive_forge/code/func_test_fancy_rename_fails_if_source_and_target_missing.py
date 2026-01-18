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
def test_fancy_rename_fails_if_source_and_target_missing(self):
    self.assertRaises((IOError, OSError), self._fancy_rename, 'missingsource', 'missingtarget')