from __future__ import annotations
import errno
import socket
import sys
from typing import Sequence
from twisted.internet import error
from twisted.trial import unittest
def assertErrnoException(self, errno: int, expectedClass: type[error.ConnectError]) -> None:
    """
        When called with a tuple with the given errno,
        L{error.getConnectError} returns an exception which is an instance of
        the expected class.
        """
    e = (errno, 'lalala')
    result = error.getConnectError(e)
    self.assertCorrectException(errno, 'lalala', result, expectedClass)