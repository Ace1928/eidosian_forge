from __future__ import annotations
from datetime import datetime as DateTime
from typing import Any, Callable, Iterator, Mapping, Optional, Union, cast
from constantly import NamedConstant
from twisted.python._tzhelper import FixedOffsetTimeZone
from twisted.python.failure import Failure
from twisted.python.reflect import safe_repr
from ._flatten import aFormatter, flatFormat
from ._interfaces import LogEvent
def eventAsText(event: LogEvent, includeTraceback: bool=True, includeTimestamp: bool=True, includeSystem: bool=True, formatTime: Callable[[float], str]=formatTime) -> str:
    """
    Format an event as text.  Optionally, attach timestamp, traceback, and
    system information.

    The full output format is:
    C{"{timeStamp} [{system}] {event}\\n{traceback}\\n"} where:

        - C{timeStamp} is the event's C{"log_time"} value formatted with
          the provided C{formatTime} callable.

        - C{system} is the event's C{"log_system"} value, if set, otherwise,
          the C{"log_namespace"} and C{"log_level"}, joined by a C{"#"}.  Each
          defaults to C{"-"} is not set.

        - C{event} is the event, as formatted by L{formatEvent}.

        - C{traceback} is the traceback if the event contains a
          C{"log_failure"} key.  In the event the original traceback cannot
          be formatted, a message indicating the failure will be substituted.

    If the event cannot be formatted, and no traceback exists, an empty string
    is returned, even if includeSystem or includeTimestamp are true.

    @param event: A logging event.
    @param includeTraceback: If true and a C{"log_failure"} key exists, append
        a traceback.
    @param includeTimestamp: If true include a formatted timestamp before the
        event.
    @param includeSystem:  If true, include the event's C{"log_system"} value.
    @param formatTime: A time formatter

    @return: A formatted string with specified options.

    @since: Twisted 18.9.0
    """
    eventText = _formatEvent(event)
    if includeTraceback and 'log_failure' in event:
        f = event['log_failure']
        traceback = _formatTraceback(f)
        eventText = '\n'.join((eventText, traceback))
    if not eventText:
        return eventText
    timeStamp = ''
    if includeTimestamp:
        timeStamp = ''.join([formatTime(cast(float, event.get('log_time', None))), ' '])
    system = ''
    if includeSystem:
        system = ''.join(['[', _formatSystem(event), ']', ' '])
    return '{timeStamp}{system}{eventText}'.format(timeStamp=timeStamp, system=system, eventText=eventText)