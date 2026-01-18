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
def cast_function(groups):
    for key, converter in cast.items():
        if key in groups:
            groups[key] = converter(groups[key])