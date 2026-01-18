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
@skipIf(platform.getType() == 'win32', 'Windows accept(2) cannot generate ENFILE')
def test_noFilesFromAccept(self):
    """
        Similar to L{test_tooManyFilesFromAccept}, but test the case where
        C{accept(2)} fails with C{ENFILE}.

        This can occur on Linux when the system has exhausted (!) its supply
        of inodes.
        """
    return self._acceptFailureTest(ENFILE)