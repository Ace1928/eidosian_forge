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
def eventFromJSON(eventText: str) -> JSONDict:
    """
    Decode a log event from JSON.

    @param eventText: The output of a previous call to L{eventAsJSON}

    @return: A reconstructed version of the log event.
    """
    return cast(JSONDict, loads(eventText, object_hook=objectLoadHook))