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
def contains_field(self, format_string, field_name):
    """
        Get the field names referenced by a format string.

        :param format_string: The logging format string.
        :returns: A list of strings with field names.
        """
    return field_name in self.get_field_names(format_string)