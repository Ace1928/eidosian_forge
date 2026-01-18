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
def getmodified(self, path):
    if self.supports_mdtm:
        _path = self.validatepath(path)
        with self._lock:
            with ftp_errors(self, path=path):
                cmd = 'MDTM ' + _encode(_path, self.ftp.encoding)
                response = self.ftp.sendcmd(cmd)
                mtime = self._parse_ftp_time(response.split()[1])
                return epoch_to_datetime(mtime)
    return super(FTPFS, self).getmodified(path)