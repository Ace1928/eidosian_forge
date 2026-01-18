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
def ansi_strip(text, readline_hints=True):
    """
    Strip ANSI escape sequences from the given string.

    :param text: The text from which ANSI escape sequences should be removed (a
                 string).
    :param readline_hints: If :data:`True` then :func:`readline_strip()` is
                           used to remove `readline hints`_ from the string.
    :returns: The text without ANSI escape sequences (a string).
    """
    pattern = '%s.*?%s' % (re.escape(ANSI_CSI), re.escape(ANSI_SGR))
    text = re.sub(pattern, '', text)
    if readline_hints:
        text = readline_strip(text)
    return text