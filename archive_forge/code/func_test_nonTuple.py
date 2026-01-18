from __future__ import annotations
import errno
import socket
import sys
from typing import Sequence
from twisted.internet import error
from twisted.trial import unittest
def test_nonTuple(self) -> None:
    """
        L{error.getConnectError} converts to a L{error.ConnectError} given
        an argument that cannot be unpacked.
        """
    e = Exception()
    result = error.getConnectError(e)
    self.assertCorrectException(None, e, result, error.ConnectError)