import json
from itertools import count
from typing import Any, Callable, Optional
from twisted.trial import unittest
from .._flatten import KeyFlattener, aFormatter, extractField, flattenEvent
from .._format import formatEvent
from .._interfaces import LogEvent
def test_formatEventFlatTrailingText(self) -> None:
    """
        L{formatEvent} will handle a flattened event with tailing text after
        a replacement field.
        """
    event = dict(log_format='test {x} trailing', x='value')
    flattenEvent(event)
    result = formatEvent(event)
    self.assertEqual(result, 'test value trailing')