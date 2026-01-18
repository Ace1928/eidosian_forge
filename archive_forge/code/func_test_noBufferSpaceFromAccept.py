import errno
import os
import socket
from unittest import skipIf
from twisted.internet import interfaces, reactor
from twisted.internet.defer import gatherResults, maybeDeferred
from twisted.internet.protocol import Protocol, ServerFactory
from twisted.internet.tcp import (
from twisted.python import log
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
def test_noBufferSpaceFromAccept(self):
    """
        Similar to L{test_tooManyFilesFromAccept}, but test the case where
        C{accept(2)} fails with C{ENOBUFS}.

        This mainly occurs on Windows and FreeBSD, but may be possible on
        Linux and other platforms as well.
        """
    return self._acceptFailureTest(ENOBUFS)