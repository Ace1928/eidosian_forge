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
def test_copy_ownership_nonesrc(self):
    """copy_ownership_from_path test with src=None."""
    open('test_file', 'w').close()
    osutils.copy_ownership_from_path('test_file')
    s = os.stat('..')
    self.assertEqual(self.path, 'test_file')
    self.assertEqual(self.uid, s.st_uid)
    self.assertEqual(self.gid, s.st_gid)