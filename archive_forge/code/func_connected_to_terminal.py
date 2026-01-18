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
def connected_to_terminal(stream=None):
    """
    Check if a stream is connected to a terminal.

    :param stream: The stream to check (a file-like object,
                   defaults to :data:`sys.stdout`).
    :returns: :data:`True` if the stream is connected to a terminal,
              :data:`False` otherwise.

    See also :func:`terminal_supports_colors()`.
    """
    stream = sys.stdout if stream is None else stream
    try:
        return stream.isatty()
    except Exception:
        return False