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
@contextmanager
def ftp_errors(fs, path=None):
    try:
        with fs._lock:
            yield
    except socket.error:
        raise errors.RemoteConnectionError(msg='unable to connect to {}'.format(fs.host))
    except EOFError:
        raise errors.RemoteConnectionError(msg='lost connection to {}'.format(fs.host))
    except error_temp as error:
        if path is not None:
            raise errors.ResourceError(path, msg="ftp error on resource '{}' ({})".format(path, error))
        else:
            raise errors.OperationFailed(msg='ftp error ({})'.format(error))
    except error_perm as error:
        code, message = _parse_ftp_error(error)
        if code == '552':
            raise errors.InsufficientStorage(path=path, msg=message)
        elif code in ('501', '550'):
            raise errors.ResourceNotFound(path=cast(str, path))
        raise errors.PermissionDenied(msg=message)