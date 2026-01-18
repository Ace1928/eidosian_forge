from __future__ import annotations
from typing import Sequence
from twisted.trial.unittest import TestCase
from twisted.web.http_headers import Headers
from twisted.web.test.requesthelper import (
def test_getRawHeadersNoDefault(self) -> None:
    """
        L{Headers.getRawHeaders} returns L{None} if the header is not found and
        no default is specified.
        """
    self.assertIsNone(Headers().getRawHeaders('test'))