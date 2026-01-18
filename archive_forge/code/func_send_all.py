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
def send_all(sock, bytes, report_activity=None):
    """Send all bytes on a socket.

    Breaks large blocks in smaller chunks to avoid buffering limitations on
    some platforms, and catches EINTR which may be thrown if the send is
    interrupted by a signal.

    This is preferred to socket.sendall(), because it avoids portability bugs
    and provides activity reporting.

    :param report_activity: Call this as bytes are read, see
        Transport._report_activity
    """
    sent_total = 0
    byte_count = len(bytes)
    view = memoryview(bytes)
    while sent_total < byte_count:
        try:
            sent = sock.send(view[sent_total:sent_total + MAX_SOCKET_CHUNK])
        except OSError as e:
            if e.args[0] in _end_of_stream_errors:
                raise errors.ConnectionReset('Error trying to write to socket', e)
            if e.args[0] != errno.EINTR:
                raise
        else:
            if sent == 0:
                raise errors.ConnectionReset('Sending to %s returned 0 bytes' % (sock,))
            sent_total += sent
            if report_activity is not None:
                report_activity(sent, 'write')