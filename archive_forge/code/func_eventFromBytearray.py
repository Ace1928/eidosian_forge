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
def eventFromBytearray(record: bytearray) -> Optional[LogEvent]:
    try:
        text = bytes(record).decode('utf-8')
    except UnicodeDecodeError:
        log.error('Unable to decode UTF-8 for JSON record: {record!r}', record=bytes(record))
        return None
    try:
        return eventFromJSON(text)
    except ValueError:
        log.error('Unable to read JSON record: {record!r}', record=bytes(record))
        return None