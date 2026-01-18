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
def envvar_to_override(self):
    if sys.platform == 'win32':
        self.overrideAttr(win32utils.ctypes, 'windll', None)
        return 'USERNAME'
    return 'LOGNAME'