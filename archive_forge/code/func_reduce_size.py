from __future__ import with_statement
import logging
import os
from textwrap import dedent
import time
import pathlib
import pydoc
import re
import functools
import traceback
import warnings
import inspect
import weakref
from datetime import timedelta
from tokenize import open as open_py_source
from . import hashing
from .func_inspect import get_func_code, get_func_name, filter_args
from .func_inspect import format_call
from .func_inspect import format_signature
from .logger import Logger, format_time, pformat
from ._store_backends import StoreBackendBase, FileSystemStoreBackend
from ._store_backends import CacheWarning  # noqa
def reduce_size(self, bytes_limit=None, items_limit=None, age_limit=None):
    """Remove cache elements to make the cache fit its limits.

        The limitation can impose that the cache size fits in ``bytes_limit``,
        that the number of cache items is no more than ``items_limit``, and
        that all files in cache are not older than ``age_limit``.

        Parameters
        ----------
        bytes_limit: int | str, optional
            Limit in bytes of the size of the cache. By default, the size of
            the cache is unlimited. When reducing the size of the cache,
            ``joblib`` keeps the most recently accessed items first. If a
            str is passed, it is converted to a number of bytes using units
            { K | M | G} for kilo, mega, giga.

        items_limit: int, optional
            Number of items to limit the cache to.  By default, the number of
            items in the cache is unlimited.  When reducing the size of the
            cache, ``joblib`` keeps the most recently accessed items first.

        age_limit: datetime.timedelta, optional
            Maximum age of items to limit the cache to.  When reducing the size
            of the cache, any items last accessed more than the given length of
            time ago are deleted.
        """
    if bytes_limit is None:
        bytes_limit = self.bytes_limit
    if self.store_backend is None:
        return
    if bytes_limit is None and items_limit is None and (age_limit is None):
        return
    self.store_backend.enforce_store_limits(bytes_limit, items_limit, age_limit)