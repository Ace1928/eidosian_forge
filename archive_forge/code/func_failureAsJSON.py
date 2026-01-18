from json import dumps, loads
from typing import IO, Any, AnyStr, Dict, Iterable, Optional, Union, cast
from uuid import UUID
from constantly import NamedConstant
from twisted.python.failure import Failure
from ._file import FileLogObserver
from ._flatten import flattenEvent
from ._interfaces import LogEvent
from ._levels import LogLevel
from ._logger import Logger
def failureAsJSON(failure: Failure) -> JSONDict:
    """
    Convert a failure to a JSON-serializable data structure.

    @param failure: A failure to serialize.

    @return: a mapping of strings to ... stuff, mostly reminiscent of
        L{Failure.__getstate__}
    """
    return dict(failure.__getstate__(), type=dict(__module__=failure.type.__module__, __name__=failure.type.__name__))