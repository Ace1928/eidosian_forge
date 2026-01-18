import json
from itertools import count
from typing import Any, Callable, Optional
from twisted.trial import unittest
from .._flatten import KeyFlattener, aFormatter, extractField, flattenEvent
from .._format import formatEvent
from .._interfaces import LogEvent
def test_extractField(self, flattenFirst: Callable[[LogEvent], LogEvent]=lambda x: x) -> None:
    """
        L{extractField} will extract a field used in the format string.

        @param flattenFirst: callable to flatten an event
        """

    class ObjectWithRepr:

        def __repr__(self) -> str:
            return 'repr'

    class Something:

        def __init__(self) -> None:
            self.number = 7
            self.object = ObjectWithRepr()

        def __getstate__(self) -> None:
            raise NotImplementedError('Just in case.')
    event = dict(log_format='{something.number} {something.object}', something=Something())
    flattened = flattenFirst(event)

    def extract(field: str) -> Any:
        return extractField(field, flattened)
    self.assertEqual(extract('something.number'), 7)
    self.assertEqual(extract('something.number!s'), '7')
    self.assertEqual(extract('something.object!s'), 'repr')