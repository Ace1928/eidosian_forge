import builtins
import contextlib
import functools
import logging
import re
import sys
import warnings
from collections import namedtuple
from inspect import isclass, iscoroutinefunction, isgeneratorfunction
from multiprocessing import current_process, get_context
from multiprocessing.context import BaseContext
from os.path import basename, splitext
from threading import current_thread
from . import _asyncio_loop, _colorama, _defaults, _filters
from ._better_exceptions import ExceptionFormatter
from ._colorizer import Colorizer
from ._contextvars import ContextVar
from ._datetime import aware_now
from ._error_interceptor import ErrorInterceptor
from ._file_sink import FileSink
from ._get_frame import get_frame
from ._handler import Handler
from ._locks_machinery import create_logger_lock
from ._recattrs import RecordException, RecordFile, RecordLevel, RecordProcess, RecordThread
from ._simple_sinks import AsyncSink, CallableSink, StandardSink, StreamSink
@contextlib.contextmanager
def contextualize(__self, **kwargs):
    """Bind attributes to the context-local ``extra`` dict while inside the ``with`` block.

        Contrary to |bind| there is no ``logger`` returned, the ``extra`` dict is modified in-place
        and updated globally. Most importantly, it uses |contextvars| which means that
        contextualized values are unique to each threads and asynchronous tasks.

        The ``extra`` dict will retrieve its initial state once the context manager is exited.

        Parameters
        ----------
        **kwargs
            Mapping between keys and values that will be added to the context-local ``extra`` dict.

        Returns
        -------
        :term:`context manager` / :term:`decorator`
            A context manager (usable as a decorator too) that will bind the attributes once entered
            and restore the initial state of the ``extra`` dict while exited.

        Examples
        --------
        >>> logger.add(sys.stderr, format="{message} | {extra}")
        1
        >>> def task():
        ...     logger.info("Processing!")
        ...
        >>> with logger.contextualize(task_id=123):
        ...     task()
        ...
        Processing! | {'task_id': 123}
        >>> logger.info("Done.")
        Done. | {}
        """
    with __self._core.lock:
        new_context = {**context.get(), **kwargs}
        token = context.set(new_context)
    try:
        yield
    finally:
        with __self._core.lock:
            context.reset(token)