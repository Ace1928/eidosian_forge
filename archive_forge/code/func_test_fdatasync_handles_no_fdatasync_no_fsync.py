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
def test_fdatasync_handles_no_fdatasync_no_fsync(self):
    self.overrideAttr(os, 'fdatasync')
    self.overrideAttr(os, 'fsync')
    self.do_fdatasync()