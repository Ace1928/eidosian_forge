from __future__ import annotations
import os
import io
import abc
import zlib
import errno
import time
import struct
import sqlite3
import threading
import pickletools
import asyncio
import inspect
import dill as pkl
import functools as ft
import contextlib as cl
import warnings
from fileio.lib.types import File, FileLike
from typing import Any, Optional, Type, Dict, Union, Tuple, TYPE_CHECKING
from lazyops.utils.pooler import ThreadPooler
from lazyops.libs.sqlcache.constants import (
from lazyops.libs.sqlcache.config import SqlCacheConfig
from lazyops.libs.sqlcache.exceptions import (
from lazyops.libs.sqlcache.utils import (
def evict(self, tag: str, retry: bool=False):
    """Remove items with matching `tag` from cache.
        Removing items is an iterative process. In each iteration, a subset of
        items is removed. Concurrent writes may occur between iterations.
        If a :exc:`Timeout` occurs, the first element of the exception's
        `args` attribute will be the number of items removed before the
        exception occurred.
        Raises :exc:`Timeout` error when database timeout occurs and `retry` is
        `False` (default).
        :param str tag: tag identifying items
        :param bool retry: retry if database timeout occurs (default False)
        :return: count of rows removed
        :raises Timeout: if database timeout occurs
        """
    select = f'SELECT rowid, filename FROM {self.table_name} WHERE tag = ? AND rowid > ? ORDER BY rowid LIMIT ?'
    args = [tag, 0, 100]
    return self._select_delete(select, args, arg_index=1, retry=retry)