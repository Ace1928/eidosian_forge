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
def get_grouped_pairs(self, format_string):
    """
        Group the results of :func:`get_pairs()` separated by whitespace.

        :param format_string: The logging format string.
        :returns: A list of lists of :class:`FormatStringToken` objects.
        """
    separated = []
    pattern = re.compile('(\\s+)')
    for token in self.get_pairs(format_string):
        if token.name:
            separated.append(token)
        else:
            separated.extend((FormatStringToken(name=None, text=text) for text in pattern.split(token.text) if text))
    current_group = []
    grouped_pairs = []
    for token in separated:
        if token.text.isspace():
            if current_group:
                grouped_pairs.append(current_group)
            grouped_pairs.append([token])
            current_group = []
        else:
            current_group.append(token)
    if current_group:
        grouped_pairs.append(current_group)
    return grouped_pairs