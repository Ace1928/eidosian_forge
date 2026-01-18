from __future__ import annotations
from typing import Sequence
from twisted.trial.unittest import TestCase
from twisted.web.http_headers import Headers
from twisted.web.test.requesthelper import (
def test_getRawHeadersDefaultValue(self) -> None:
    """
        L{Headers.getRawHeaders} returns the specified default value when no
        header is found.
        """
    h = Headers()
    default = object()
    self.assertIdentical(h.getRawHeaders('test', default), default)
    self.assertIdentical(h.getRawHeaders('test', None), None)
    self.assertEqual(h.getRawHeaders('test', [None]), [None])
    self.assertEqual(h.getRawHeaders('test', ['☃']), ['☃'])