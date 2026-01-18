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
def call_and_shelve(self, *args, **kwargs):
    """Call wrapped function, cache result and return a reference.

        This method returns a reference to the cached result instead of the
        result itself. The reference object is small and pickeable, allowing
        to send or store it easily. Call .get() on reference object to get
        result.

        Returns
        -------
        cached_result: MemorizedResult or NotMemorizedResult
            reference to the value returned by the wrapped function. The
            class "NotMemorizedResult" is used when there is no cache
            activated (e.g. location=None in Memory).
        """
    _, args_id, metadata = self._cached_call(args, kwargs, shelving=True)
    return MemorizedResult(self.store_backend, self.func, args_id, metadata=metadata, verbose=self._verbose - 1, timestamp=self.timestamp)