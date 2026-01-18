from __future__ import annotations
from typing import Sequence
from twisted.trial.unittest import TestCase
from twisted.web.http_headers import Headers
from twisted.web.test.requesthelper import (
def test_nameNotEncodable(self) -> None:
    """
        Passing L{str} to any function that takes a header name will encode
        said header name as ISO-8859-1, and if it cannot be encoded, it will
        raise a L{UnicodeDecodeError}.
        """
    h = Headers()
    with self.assertRaises(UnicodeEncodeError):
        h.setRawHeaders('☃', ['val'])
    with self.assertRaises(UnicodeEncodeError):
        h.hasHeader('☃')