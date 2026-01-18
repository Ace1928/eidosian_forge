from __future__ import annotations
from typing import Sequence
from twisted.trial.unittest import TestCase
from twisted.web.http_headers import Headers
from twisted.web.test.requesthelper import (
def test_getRawHeadersWithDefaultMatchingValue(self) -> None:
    """
        If the object passed as the value list to L{Headers.setRawHeaders}
        is later passed as a default to L{Headers.getRawHeaders}, the
        result nevertheless contains decoded values.
        """
    h = Headers()
    default = [b'value']
    h.setRawHeaders(b'key', default)
    self.assertIsInstance(h.getRawHeaders('key', default)[0], str)
    self.assertEqual(h.getRawHeaders('key', default), ['value'])