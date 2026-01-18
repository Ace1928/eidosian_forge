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
def test_BRZ_COLUMNS_0_no_limit(self):
    self.overrideEnv('BRZ_COLUMNS', '0')
    self.assertEqual(None, osutils.terminal_width())