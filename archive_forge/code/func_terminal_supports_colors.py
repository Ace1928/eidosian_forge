import codecs
import numbers
import os
import platform
import re
import subprocess
import sys
from humanfriendly.compat import coerce_string, is_unicode, on_windows, which
from humanfriendly.decorators import cached
from humanfriendly.deprecation import define_aliases
from humanfriendly.text import concatenate, format
from humanfriendly.usage import format_usage
def terminal_supports_colors(stream=None):
    """
    Check if a stream is connected to a terminal that supports ANSI escape sequences.

    :param stream: The stream to check (a file-like object,
                   defaults to :data:`sys.stdout`).
    :returns: :data:`True` if the terminal supports ANSI escape sequences,
              :data:`False` otherwise.

    This function was originally inspired by the implementation of
    `django.core.management.color.supports_color()
    <https://github.com/django/django/blob/master/django/core/management/color.py>`_
    but has since evolved significantly.
    """
    if on_windows():
        have_ansicon = 'ANSICON' in os.environ
        have_colorama = 'colorama' in sys.modules
        have_native_support = have_windows_native_ansi_support()
        if not (have_ansicon or have_colorama or have_native_support):
            return False
    return connected_to_terminal(stream)