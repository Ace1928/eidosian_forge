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
def test_more_than_segment_size(self):
    output = BytesIO()
    osutils.pump_string_file(b'123456789', output, 2)
    self.assertEqual(b'123456789', output.getvalue())