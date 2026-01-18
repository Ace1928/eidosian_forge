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
def test_local_time_offset(self):
    """Test that local_time_offset() returns a sane value."""
    offset = osutils.local_time_offset()
    self.assertTrue(isinstance(offset, int))
    eighteen_hours = 18 * 3600
    self.assertTrue(-eighteen_hours < offset < eighteen_hours)