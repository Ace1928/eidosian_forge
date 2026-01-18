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
def test_env_disable(self):
    """Make sure ANSI escape sequences can be disabled using ``$NO_COLOR``."""
    with PatchedItem(os.environ, 'NO_COLOR', 'I like monochrome'):
        with CaptureOutput() as capturer:
            subprocess.check_call([sys.executable, '-c', ';'.join(['import coloredlogs, logging', 'coloredlogs.install()', "logging.info('Hello world')"])])
            output = capturer.get_text()
            assert ANSI_CSI not in output