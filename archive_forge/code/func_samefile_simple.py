from __future__ import annotations
import errno
import importlib.util
import os
import socket
import sys
import warnings
from contextlib import contextmanager
from typing import Any, Generator, NewType, Sequence
from urllib.parse import (
from urllib.parse import (
from urllib.request import pathname2url as _pathname2url
from _frozen_importlib_external import _NamespacePath
from jupyter_core.utils import ensure_async as _ensure_async
from packaging.version import Version
from tornado.httpclient import AsyncHTTPClient, HTTPClient, HTTPRequest, HTTPResponse
from tornado.netutil import Resolver
def samefile_simple(path: str, other_path: str) -> bool:
    """
    Fill in for os.path.samefile when it is unavailable (Windows+py2).

    Do a case-insensitive string comparison in this case
    plus comparing the full stat result (including times)
    because Windows + py2 doesn't support the stat fields
    needed for identifying if it's the same file (st_ino, st_dev).

    Only to be used if os.path.samefile is not available.

    Parameters
    ----------
    path : str
        representing a path to a file
    other_path : str
        representing a path to another file

    Returns
    -------
    same:   Boolean that is True if both path and other path are the same
    """
    path_stat = os.stat(path)
    other_path_stat = os.stat(other_path)
    return path.lower() == other_path.lower() and path_stat == other_path_stat