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
def get_logger_tree(self):
    """Create and return a tree of loggers."""
    root = logging.getLogger()
    parent_name = random_string()
    parent = logging.getLogger(parent_name)
    child_name = '%s.%s' % (parent_name, random_string())
    child = logging.getLogger(child_name)
    grand_child_name = '%s.%s' % (child_name, random_string())
    grand_child = logging.getLogger(grand_child_name)
    return (root, parent, child, grand_child)