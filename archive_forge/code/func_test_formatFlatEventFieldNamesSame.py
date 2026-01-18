import json
from itertools import count
from typing import Any, Callable, Optional
from twisted.trial import unittest
from .._flatten import KeyFlattener, aFormatter, extractField, flattenEvent
from .._format import formatEvent
from .._interfaces import LogEvent
def test_formatFlatEventFieldNamesSame(self) -> None:
    """
        The same format field used twice in one event is rendered twice.
        """
    self._test_formatFlatEvent_fieldNamesSame()