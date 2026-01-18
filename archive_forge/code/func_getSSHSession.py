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
def getSSHSession(self, register_adapters=True):
    """
        Return a new SSH session.
        """
    return session.SSHSession(remoteWindow=500, remoteMaxPacket=100, conn=StubConnection(), avatar=StubAvatar())