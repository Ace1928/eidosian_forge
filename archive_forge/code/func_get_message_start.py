import contextlib
import errno
import logging
import logging.handlers
import os
import sys
import threading
from dataclasses import dataclass
from io import TextIOWrapper
from logging import Filter
from typing import Any, ClassVar, Generator, List, Optional, TextIO, Type
from pip._vendor.rich.console import (
from pip._vendor.rich.highlighter import NullHighlighter
from pip._vendor.rich.logging import RichHandler
from pip._vendor.rich.segment import Segment
from pip._vendor.rich.style import Style
from pip._internal.utils._log import VERBOSE, getLogger
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.deprecation import DEPRECATION_MSG_PREFIX
from pip._internal.utils.misc import ensure_dir
def get_message_start(self, formatted: str, levelno: int) -> str:
    """
        Return the start of the formatted log message (not counting the
        prefix to add to each line).
        """
    if levelno < logging.WARNING:
        return ''
    if formatted.startswith(DEPRECATION_MSG_PREFIX):
        return ''
    if levelno < logging.ERROR:
        return 'WARNING: '
    return 'ERROR: '