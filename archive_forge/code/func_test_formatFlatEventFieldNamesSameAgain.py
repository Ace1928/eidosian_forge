import json
from itertools import count
from typing import Any, Callable, Optional
from twisted.trial import unittest
from .._flatten import KeyFlattener, aFormatter, extractField, flattenEvent
from .._format import formatEvent
from .._interfaces import LogEvent
def test_formatFlatEventFieldNamesSameAgain(self) -> None:
    """
        The same event flattened twice gives the same (already rendered)
        result.
        """
    event = self._test_formatFlatEvent_fieldNamesSame()
    self._test_formatFlatEvent_fieldNamesSame(event)