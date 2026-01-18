import os
import signal
import struct
import sys
from unittest import skipIf
from zope.interface import implementer
from twisted.internet import defer, error, protocol
from twisted.internet.address import IPv4Address
from twisted.internet.error import ProcessDone, ProcessTerminated
from twisted.python import components, failure
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.python.test.test_components import RegistryUsingMixin
from twisted.trial.unittest import TestCase
def execCommand(self, pp, command):
    """
        If the command is 'true', store the command, the process protocol, and
        the transport we connect to the process protocol.  Otherwise, just
        store the command and raise an error.
        """
    self.execCommandLine = command
    if command == b'success':
        self.execProtocol = pp
    elif command[:6] == b'repeat':
        self.execProtocol = pp
        self.execTransport = EchoTransport(pp)
        pp.outReceived(command[7:])
    else:
        raise RuntimeError('not getting a command')