import collections
import logging
import os
import re
import socket
import sys
from humanfriendly import coerce_boolean
from humanfriendly.compat import coerce_string, is_string, on_windows
from humanfriendly.terminal import ANSI_COLOR_CODES, ansi_wrap, enable_ansi_support, terminal_supports_colors
from humanfriendly.text import format, split
def adjust_level(logger, level):
    """
    Increase a logger's verbosity up to the requested level.

    :param logger: The logger to change (a :class:`~logging.Logger` object).
    :param level: The log level to enable (a string or number).

    This function is used by functions like :func:`install()`,
    :func:`increase_verbosity()` and :func:`.enable_system_logging()` to adjust
    a logger's level so that log messages up to the requested log level are
    propagated to the configured output handler(s).

    It uses :func:`logging.Logger.getEffectiveLevel()` to check whether
    `logger` propagates or swallows log messages of the requested `level` and
    sets the logger's level to the requested level if it would otherwise
    swallow log messages.

    Effectively this function will "widen the scope of logging" when asked to
    do so but it will never "narrow the scope of logging". This is because I am
    convinced that filtering of log messages should (primarily) be decided by
    handlers.
    """
    level = level_to_number(level)
    if logger.getEffectiveLevel() > level:
        logger.setLevel(level)