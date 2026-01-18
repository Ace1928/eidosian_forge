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
def test_deep_relpath(self):
    cwd = osutils.getcwd()
    subdir = cwd + '/sub/subsubdir'
    self.assertEqual('sub/subsubdir', osutils.relpath(cwd, subdir))