import json
from itertools import count
from typing import Any, Callable, Optional
from twisted.trial import unittest
from .._flatten import KeyFlattener, aFormatter, extractField, flattenEvent
from .._format import formatEvent
from .._interfaces import LogEvent
def test_extractFieldFlattenFirst(self) -> None:
    """
        L{extractField} behaves identically if the event is explicitly
        flattened first.
        """

    def flattened(event: LogEvent) -> LogEvent:
        flattenEvent(event)
        return event
    self.test_extractField(flattened)