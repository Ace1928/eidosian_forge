from __future__ import annotations
import fnmatch as _fnmatch
import functools
import io
import logging
import os
import platform
import re
import sys
import textwrap
import tokenize
from typing import NamedTuple
from typing import Pattern
from typing import Sequence
from flake8 import exceptions
def matches_filename(path: str, patterns: Sequence[str], log_message: str, logger: logging.Logger) -> bool:
    """Use fnmatch to discern if a path exists in patterns.

    :param path:
        The path to the file under question
    :param patterns:
        The patterns to match the path against.
    :param log_message:
        The message used for logging purposes.
    :returns:
        True if path matches patterns, False otherwise
    """
    if not patterns:
        return False
    basename = os.path.basename(path)
    if basename not in {'.', '..'} and fnmatch(basename, patterns):
        logger.debug(log_message, {'path': basename, 'whether': ''})
        return True
    absolute_path = os.path.abspath(path)
    match = fnmatch(absolute_path, patterns)
    logger.debug(log_message, {'path': absolute_path, 'whether': '' if match else 'not '})
    return match