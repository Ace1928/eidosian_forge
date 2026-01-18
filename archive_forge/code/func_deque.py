import contextlib as cl
import functools
import itertools as it
import operator
import os.path as op
import sqlite3
import tempfile
import time
from .core import DEFAULT_SETTINGS, ENOVAL, Cache, Disk, Timeout
from .persistent import Deque, Index
def deque(self, name, maxlen=None):
    """Return Deque with given `name` in subdirectory.

        >>> cache = FanoutCache()
        >>> deque = cache.deque('test')
        >>> deque.extend('abc')
        >>> deque.popleft()
        'a'
        >>> deque.pop()
        'c'
        >>> len(deque)
        1

        :param str name: subdirectory name for Deque
        :param maxlen: max length (default None, no max)
        :return: Deque with given name

        """
    _deques = self._deques
    try:
        return _deques[name]
    except KeyError:
        parts = name.split('/')
        directory = op.join(self._directory, 'deque', *parts)
        cache = Cache(directory=directory, disk=self._disk, eviction_policy='none')
        deque = Deque.fromcache(cache, maxlen=maxlen)
        _deques[name] = deque
        return deque