import logging
import logging.handlers
import os
import socket
import sys
from humanfriendly import coerce_boolean
from humanfriendly.compat import on_macos, on_windows
from coloredlogs import (
def match_syslog_handler(handler):
    """
    Identify system logging handlers.

    :param handler: The :class:`~logging.Handler` class to check.
    :returns: :data:`True` if the handler is a
              :class:`~logging.handlers.SysLogHandler`,
              :data:`False` otherwise.

    This function can be used as a callback for :func:`.find_handler()`.
    """
    return isinstance(handler, logging.handlers.SysLogHandler)