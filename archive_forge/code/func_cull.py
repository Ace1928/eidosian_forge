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
def cull(self, retry: bool=False):
    """Cull items from cache until volume is less than size limit.
        Removing items is an iterative process. In each iteration, a subset of
        items is removed. Concurrent writes may occur between iterations.
        If a :exc:`Timeout` occurs, the first element of the exception's
        `args` attribute will be the number of items removed before the
        exception occurred.
        Raises :exc:`Timeout` error when database timeout occurs and `retry` is
        `False` (default).
        :param bool retry: retry if database timeout occurs (default False)
        :return: count of items removed
        :raises Timeout: if database timeout occurs
        """
    now = time.time()
    count = self.expire(now)
    select_policy: str = self.eviction_policy['cull']
    if select_policy is None:
        return 0
    select_filename = select_policy.format(fields='filename', now=now)
    try:
        while self.volume() > self.config.size_limit:
            with self._transact(retry) as (sql, cleanup):
                rows = sql(select_filename, (10,)).fetchall()
                if not rows:
                    break
                count += len(rows)
                delete = f'DELETE FROM {self.table_name} WHERE rowid IN ({select_policy.format(fields='rowid', now=now)})'
                sql(delete, (10,))
                for filename, in rows:
                    cleanup(filename)
    except SqlTimeout:
        raise SqlTimeout(count) from None
    return count