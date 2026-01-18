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
def test_defaults_to_BRZ_COLUMNS(self):
    self.assertNotEqual('12', os.environ['BRZ_COLUMNS'])
    self.overrideEnv('BRZ_COLUMNS', '12')
    self.assertEqual(12, osutils.terminal_width())