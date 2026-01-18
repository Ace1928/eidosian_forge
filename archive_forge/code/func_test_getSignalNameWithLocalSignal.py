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
@skipIf(not hasattr(signal, 'SIGALRM'), 'Not all signals available')
def test_getSignalNameWithLocalSignal(self):
    """
        If there are signals in the signal module which aren't in the SSH RFC,
        we map their name to [signal name]@[platform].
        """
    signal.SIGTwistedTest = signal.NSIG + 1
    self.pp._signalValuesToNames = None
    self.assertEqual(self.pp._getSignalName(signal.SIGTwistedTest), 'SIGTwistedTest@' + sys.platform)