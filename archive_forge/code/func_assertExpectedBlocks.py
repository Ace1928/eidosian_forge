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
def assertExpectedBlocks(self, expected, result):
    self.assertEqual(expected, [(dirinfo, [line[0:3] for line in block]) for dirinfo, block in result])