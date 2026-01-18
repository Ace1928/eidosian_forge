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
def test_empty_conversion(self):
    """
        Test that conversion of empty output produces no HTML.

        This test was added because I found that ``coloredlogs --convert`` when
        used in a cron job could cause cron to send out what appeared to be
        empty emails. On more careful inspection the body of those emails was
        ``<code></code>``. By not emitting the wrapper element when no other
        HTML is generated, cron will not send out an email.
        """
    output = main('coloredlogs', '--convert', 'true', capture=True)
    assert not output.strip()