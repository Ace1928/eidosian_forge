from __future__ import annotations
from random import randrange
from typing import Any, Callable, TextIO, TypeVar
from typing_extensions import ParamSpec
from twisted.internet import interfaces, utils
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.lockfile import FilesystemLock
def excInfoOrFailureToExcInfo(err):
    """
    Coerce a Failure to an _exc_info, if err is a Failure.

    @param err: Either a tuple such as returned by L{sys.exc_info} or a
        L{Failure} object.
    @return: A tuple like the one returned by L{sys.exc_info}. e.g.
        C{exception_type, exception_object, traceback_object}.
    """
    if isinstance(err, Failure):
        err = (err.type, err.value, err.getTracebackObject())
    return err