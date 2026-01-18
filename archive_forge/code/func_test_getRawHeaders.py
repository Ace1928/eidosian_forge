from __future__ import annotations
from typing import Sequence
from twisted.trial.unittest import TestCase
from twisted.web.http_headers import Headers
from twisted.web.test.requesthelper import (
def test_getRawHeaders(self) -> None:
    """
        L{Headers.getRawHeaders} returns the values which have been set for a
        given header.
        """
    h = Headers()
    h.setRawHeaders('testá', ['lemur'])
    self.assertEqual(h.getRawHeaders('testá'), ['lemur'])
    self.assertEqual(h.getRawHeaders('Testá'), ['lemur'])
    self.assertEqual(h.getRawHeaders(b'test\xe1'), [b'lemur'])
    self.assertEqual(h.getRawHeaders(b'Test\xe1'), [b'lemur'])