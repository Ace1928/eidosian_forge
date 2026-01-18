from __future__ import annotations
from typing import Sequence
from twisted.trial.unittest import TestCase
from twisted.web.http_headers import Headers
from twisted.web.test.requesthelper import (
def test_addRawHeader(self) -> None:
    """
        L{Headers.addRawHeader} accepts mixed L{str} and L{bytes}.
        """
    h = Headers()
    h.addRawHeader(b'bytes', 'str')
    h.addRawHeader('str', b'bytes')
    self.assertEqual(h.getRawHeaders(b'Bytes'), [b'str'])
    self.assertEqual(h.getRawHeaders('Str'), ['bytes'])