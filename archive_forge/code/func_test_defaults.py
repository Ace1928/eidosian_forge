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
def test_defaults(self):
    """Verifies that the default arguments will read to EOF -- this
        test verifies that any existing usages of pumpfile will not be broken
        with this new version."""
    from_file = file_utils.FakeReadFile(self.test_data)
    to_file = BytesIO()
    osutils.pumpfile(from_file, to_file)
    response_data = to_file.getvalue()
    if response_data != self.test_data:
        message = 'Data not equal.  Expected %d bytes, received %d.'
        self.fail(message % (len(response_data), self.test_data_len))