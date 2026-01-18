from __future__ import annotations
from typing import Sequence
from twisted.trial.unittest import TestCase
from twisted.web.http_headers import Headers
from twisted.web.test.requesthelper import (
def test_headersComparison(self) -> None:
    """
        A L{Headers} instance compares equal to itself and to another
        L{Headers} instance with the same values.
        """
    first = Headers()
    first.setRawHeaders('fooá', ['panda'])
    second = Headers()
    second.setRawHeaders('fooá', ['panda'])
    third = Headers()
    third.setRawHeaders('fooá', ['lemur', 'panda'])
    self.assertEqual(first, first)
    self.assertEqual(first, second)
    self.assertNotEqual(first, third)
    firstBytes = Headers()
    firstBytes.setRawHeaders(b'foo\xe1', [b'panda'])
    secondBytes = Headers()
    secondBytes.setRawHeaders(b'foo\xe1', [b'panda'])
    thirdBytes = Headers()
    thirdBytes.setRawHeaders(b'foo\xe1', [b'lemur', 'panda'])
    self.assertEqual(first, firstBytes)
    self.assertEqual(second, secondBytes)
    self.assertEqual(third, thirdBytes)