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
def peekitem(self, last: bool=True, expire_time: bool=False, tag: bool=False, retry: bool=False):
    """Peek at key and value item pair in cache based on iteration order.
        Expired items are deleted from cache. Operation is atomic. Concurrent
        operations will be serialized.
        Raises :exc:`Timeout` error when database timeout occurs and `retry` is
        `False` (default).
        >>> cache = Store()
        >>> for num, letter in enumerate('abc'):
        ...     cache[letter] = num
        >>> cache.peekitem()
        ('c', 2)
        >>> cache.peekitem(last=False)
        ('a', 0)
        :param bool last: last item in iteration order (default True)
        :param bool expire_time: if True, return expire_time in tuple
            (default False)
        :param bool tag: if True, return tag in tuple (default False)
        :param bool retry: retry if database timeout occurs (default False)
        :return: key and value item pair
        :raises KeyError: if cache is empty
        :raises Timeout: if database timeout occurs
        """
    order = ('ASC', 'DESC')
    select = f'SELECT rowid, key, raw, expire_time, tag, mode, filename, value FROM {self.table_name} ORDER BY rowid %s LIMIT 1' % order[last]
    while True:
        while True:
            with self._transact(retry) as (sql, cleanup):
                rows = sql(select).fetchall()
                if not rows:
                    raise KeyError('dictionary is empty')
                (rowid, db_key, raw, db_expire, db_tag, mode, name, db_value), = rows
                if db_expire is not None and db_expire < time.time():
                    sql(f'DELETE FROM {self.table_name} WHERE rowid = ?', (rowid,))
                    cleanup(name)
                else:
                    break
        key = self._medium.get(db_key, raw)
        try:
            value = self._medium.fetch(mode, name, db_value, False)
        except IOError as error:
            if error.errno == errno.ENOENT:
                continue
            raise
        break
    if expire_time and tag:
        return ((key, value), db_expire, db_tag)
    elif expire_time:
        return ((key, value), db_expire)
    elif tag:
        return ((key, value), db_tag)
    return (key, value)