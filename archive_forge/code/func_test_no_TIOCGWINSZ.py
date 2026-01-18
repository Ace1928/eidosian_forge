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
def test_no_TIOCGWINSZ(self):
    self.requireFeature(term_ios_feature)
    termios = term_ios_feature.module
    try:
        termios.TIOCGWINSZ
    except AttributeError:
        pass
    else:
        self.overrideAttr(termios, 'TIOCGWINSZ')
        del termios.TIOCGWINSZ
    self.overrideEnv('BRZ_COLUMNS', None)
    self.overrideEnv('COLUMNS', None)
    osutils.terminal_width()