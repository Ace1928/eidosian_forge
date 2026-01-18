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
def readline_wrap(expr):
    """
    Wrap an ANSI escape sequence in `readline hints`_.

    :param text: The text with the escape sequence to wrap (a string).
    :returns: The wrapped text.

    .. _readline hints: http://superuser.com/a/301355
    """
    return '\x01' + expr + '\x02'