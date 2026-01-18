import json
from itertools import count
from typing import Any, Callable, Optional
from twisted.trial import unittest
from .._flatten import KeyFlattener, aFormatter, extractField, flattenEvent
from .._format import formatEvent
from .._interfaces import LogEvent
def test_formatFlatEvent(self) -> None:
    """
        L{flattenEvent} will "flatten" an event so that, if scrubbed of all but
        serializable objects, it will preserve all necessary data to be
        formatted once serialized.  When presented with an event thusly
        flattened, L{formatEvent} will produce the same output.
        """
    counter = count()

    class Ephemeral:
        attribute = 'value'
    event1 = dict(log_format='callable: {callme()} attribute: {object.attribute} numrepr: {number!r} numstr: {number!s} strrepr: {string!r} unistr: {unistr!s}', callme=lambda: next(counter), object=Ephemeral(), number=7, string='hello', unistr='ö')
    flattenEvent(event1)
    event2 = dict(event1)
    del event2['callme']
    del event2['object']
    event3 = json.loads(json.dumps(event2))
    self.assertEqual(formatEvent(event3), "callable: 0 attribute: value numrepr: 7 numstr: 7 strrepr: 'hello' unistr: ö")