from __future__ import annotations
from datetime import datetime as DateTime
from typing import Any, Callable, Iterator, Mapping, Optional, Union, cast
from constantly import NamedConstant
from twisted.python._tzhelper import FixedOffsetTimeZone
from twisted.python.failure import Failure
from twisted.python.reflect import safe_repr
from ._flatten import aFormatter, flatFormat
from ._interfaces import LogEvent
def formatEvent(event: LogEvent) -> str:
    """
    Formats an event as text, using the format in C{event["log_format"]}.

    This implementation should never raise an exception; if the formatting
    cannot be done, the returned string will describe the event generically so
    that a useful message is emitted regardless.

    @param event: A logging event.

    @return: A formatted string.
    """
    return eventAsText(event, includeTraceback=False, includeTimestamp=False, includeSystem=False)