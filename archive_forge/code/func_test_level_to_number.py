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
def test_level_to_number(self):
    """Make sure :func:`level_to_number()` works as intended."""
    assert level_to_number('debug') == logging.DEBUG
    assert level_to_number('info') == logging.INFO
    assert level_to_number('warning') == logging.WARNING
    assert level_to_number('error') == logging.ERROR
    assert level_to_number('fatal') == logging.FATAL
    assert level_to_number('bogus-level') == logging.INFO