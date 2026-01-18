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
@skipIf(platform.getType() == 'win32', 'Windows requires an unacceptably large amount of resources to provoke this behavior in the naive manner.')
def test_acceptOutOfFiles(self):
    """
        Test that the platform accept(2) call fails with either L{EMFILE} or
        L{ENOBUFS} when there are too many file descriptors open.
        """
    port = self.socket()
    port.bind(('127.0.0.1', 0))
    serverPortNumber = port.getsockname()[1]
    port.listen(5)
    client = self.socket()
    client.setblocking(False)
    for i in range(self.socketLimit):
        try:
            self.socket()
        except OSError as e:
            if e.args[0] in (EMFILE, ENOBUFS):
                break
            else:
                raise
    else:
        self.fail('Could provoke neither EMFILE nor ENOBUFS from platform.')
    self.assertIn(client.connect_ex(('127.0.0.1', serverPortNumber)), (0, EINPROGRESS))
    exc = self.assertRaises(socket.error, port.accept)
    self.assertIn(exc.args[0], (EMFILE, ENOBUFS))