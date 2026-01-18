from __future__ import print_function, unicode_literals
import typing
import array
import calendar
import datetime
import io
import itertools
import socket
import threading
from collections import OrderedDict
from contextlib import contextmanager
from ftplib import FTP
from typing import cast
from ftplib import error_perm, error_temp
from six import PY2, raise_from, text_type
from . import _ftp_parse as ftp_parse
from . import errors
from .base import FS
from .constants import DEFAULT_CHUNK_SIZE
from .enums import ResourceType, Seek
from .info import Info
from .iotools import line_iterator
from .mode import Mode
from .path import abspath, basename, dirname, normpath, split
from .time import epoch_to_datetime
@property
def ftp_url(self):
    """Get the FTP url this filesystem will open."""
    if self.port == 21:
        _host_part = self.host
    else:
        _host_part = '{}:{}'.format(self.host, self.port)
    if self.user == 'anonymous' or self.user is None:
        _user_part = ''
    else:
        _user_part = '{}:{}@'.format(self.user, self.passwd)
    scheme = 'ftps' if self.tls else 'ftp'
    url = '{}://{}{}'.format(scheme, _user_part, _host_part)
    return url