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
def test_at_root_drive(self):
    if sys.platform != 'win32':
        raise tests.TestNotApplicable('we can only test drive-letter relative paths on Windows where we have drive letters.')
    self.assertRelpath('foo', 'C:/', 'C:/foo')
    self.assertRelpath('foo', 'X:/', 'X:/foo')
    self.assertRelpath('foo', 'X:/', 'X://foo')