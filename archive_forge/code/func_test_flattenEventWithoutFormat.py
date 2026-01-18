import json
from itertools import count
from typing import Any, Callable, Optional
from twisted.trial import unittest
from .._flatten import KeyFlattener, aFormatter, extractField, flattenEvent
from .._format import formatEvent
from .._interfaces import LogEvent
def test_flattenEventWithoutFormat(self) -> None:
    """
        L{flattenEvent} will do nothing to an event with no format string.
        """
    inputEvent = {'a': 'b', 'c': 1}
    flattenEvent(inputEvent)
    self.assertEqual(inputEvent, {'a': 'b', 'c': 1})