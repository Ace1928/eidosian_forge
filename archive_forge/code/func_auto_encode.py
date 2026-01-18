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
def auto_encode(stream, text, *args, **kw):
    """
    Reliably write Unicode strings to the terminal.

    :param stream: The file-like object to write to (a value like
                   :data:`sys.stdout` or :data:`sys.stderr`).
    :param text: The text to write to the stream (a string).
    :param args: Refer to :func:`~humanfriendly.text.format()`.
    :param kw: Refer to :func:`~humanfriendly.text.format()`.

    Renders the text using :func:`~humanfriendly.text.format()` and writes it
    to the given stream. If an :exc:`~exceptions.UnicodeEncodeError` is
    encountered in doing so, the text is encoded using :data:`DEFAULT_ENCODING`
    and the write is retried. The reasoning behind this rather blunt approach
    is that it's preferable to get output on the command line in the wrong
    encoding then to have the Python program blow up with a
    :exc:`~exceptions.UnicodeEncodeError` exception.
    """
    text = format(text, *args, **kw)
    try:
        stream.write(text)
    except UnicodeEncodeError:
        stream.write(codecs.encode(text, DEFAULT_ENCODING))