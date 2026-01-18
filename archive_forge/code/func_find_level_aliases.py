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
def find_level_aliases():
    """
    Find log level names which are aliases of each other.

    :returns: A dictionary that maps aliases to their canonical name.

    .. note:: Canonical names are chosen to be the alias with the longest
              string length so that e.g. ``WARN`` is an alias for ``WARNING``
              instead of the other way around.

    Here's what the result looks like by default (when
    no custom levels or level names have been defined):

    >>> from coloredlogs import find_level_aliases
    >>> find_level_aliases()
    {'WARN': 'WARNING', 'FATAL': 'CRITICAL'}
    """
    mapping = collections.defaultdict(list)
    for name, value in find_defined_levels().items():
        mapping[value].append(name)
    aliases = {}
    for value, names in mapping.items():
        if len(names) > 1:
            names = sorted(names, key=lambda n: len(n))
            canonical_name = names.pop()
            for alias in names:
                aliases[alias] = canonical_name
    return aliases