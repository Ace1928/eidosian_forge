import contextlib
import logging
import logging.handlers
import os
import re
import subprocess
import sys
import tempfile
from humanfriendly.compat import StringIO
from humanfriendly.terminal import ANSI_COLOR_CODES, ANSI_CSI, ansi_style, ansi_wrap
from humanfriendly.testing import PatchedAttribute, PatchedItem, TestCase, retry
from humanfriendly.text import format, random_string
import coloredlogs
import coloredlogs.cli
from coloredlogs import (
from coloredlogs.demo import demonstrate_colored_logging
from coloredlogs.syslog import SystemLogging, is_syslog_supported, match_syslog_handler
from coloredlogs.converter import (
from capturer import CaptureOutput
from verboselogs import VerboseLogger
def test_auto_install(self):
    """Test :func:`coloredlogs.auto_install()`."""
    needle = random_string()
    command_line = [sys.executable, '-c', 'import logging; logging.info(%r)' % needle]
    with CaptureOutput() as capturer:
        os.environ['COLOREDLOGS_AUTO_INSTALL'] = 'false'
        subprocess.check_call(command_line)
        output = capturer.get_text()
    assert needle not in output
    with CaptureOutput() as capturer:
        os.environ['COLOREDLOGS_AUTO_INSTALL'] = 'true'
        subprocess.check_call(command_line)
        output = capturer.get_text()
    assert needle in output