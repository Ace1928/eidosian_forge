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
def level_to_number(value):
    """
    Coerce a logging level name to a number.

    :param value: A logging level (integer or string).
    :returns: The number of the log level (an integer).

    This function translates log level names into their numeric values..
    """
    if is_string(value):
        try:
            defined_levels = find_defined_levels()
            value = defined_levels[value.upper()]
        except KeyError:
            value = DEFAULT_LOG_LEVEL
    return value