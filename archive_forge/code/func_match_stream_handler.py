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
def match_stream_handler(handler, streams=[]):
    """
    Identify stream handlers writing to the given streams(s).

    :param handler: The :class:`~logging.Handler` class to check.
    :param streams: A sequence of streams to match (defaults to matching
                    :data:`~sys.stdout` and :data:`~sys.stderr`).
    :returns: :data:`True` if the handler is a :class:`~logging.StreamHandler`
              logging to the given stream(s), :data:`False` otherwise.

    This function can be used as a callback for :func:`find_handler()`.
    """
    return isinstance(handler, logging.StreamHandler) and getattr(handler, 'stream') in (streams or (sys.stdout, sys.stderr))