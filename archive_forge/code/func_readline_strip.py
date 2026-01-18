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
def readline_strip(expr):
    """
    Remove `readline hints`_ from a string.

    :param text: The text to strip (a string).
    :returns: The stripped text.
    """
    return expr.replace('\x01', '').replace('\x02', '')