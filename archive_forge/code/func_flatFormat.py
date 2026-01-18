from collections import defaultdict
from string import Formatter
from typing import Any, Dict, Optional
from ._interfaces import LogEvent
def flatFormat(event: LogEvent) -> str:
    """
    Format an event which has been flattened with L{flattenEvent}.

    @param event: A logging event.

    @return: A formatted string.
    """
    fieldValues = event['log_flattened']
    keyFlattener = KeyFlattener()
    s = []
    for literalText, fieldName, formatSpec, conversion in aFormatter.parse(event['log_format']):
        s.append(literalText)
        if fieldName is not None:
            key = keyFlattener.flatKey(fieldName, formatSpec, conversion or 's')
            s.append(str(fieldValues[key]))
    return ''.join(s)