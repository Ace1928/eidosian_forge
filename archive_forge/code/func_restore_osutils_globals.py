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
def restore_osutils_globals(self):
    osutils._terminal_size_state = self._orig_terminal_size_state
    osutils._first_terminal_size = self._orig_first_terminal_size