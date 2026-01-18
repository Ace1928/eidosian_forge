import codecs
import errno
import os
import re
import stat
import sys
import time
from functools import partial
from typing import Dict, List
from .lazy_import import lazy_import
import locale
import ntpath
import posixpath
import select
import shutil
from shutil import rmtree
import socket
import subprocess
import unicodedata
from breezy import (
from breezy.i18n import gettext
from hashlib import md5
from hashlib import sha1 as sha
import breezy
from . import errors
def read_bytes_from_socket(sock, report_activity=None, max_read_size=MAX_SOCKET_CHUNK):
    """Read up to max_read_size of bytes from sock and notify of progress.

    Translates "Connection reset by peer" into file-like EOF (return an
    empty string rather than raise an error), and repeats the recv if
    interrupted by a signal.
    """
    while True:
        try:
            data = sock.recv(max_read_size)
        except OSError as e:
            eno = e.args[0]
            if eno in _end_of_stream_errors:
                return b''
            elif eno == errno.EINTR:
                continue
            raise
        else:
            if report_activity is not None:
                report_activity(len(data), 'read')
            return data