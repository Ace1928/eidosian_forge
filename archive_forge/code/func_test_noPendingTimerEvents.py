import errno
import sys
import time
from array import array
from socket import AF_INET, AF_INET6, SOCK_STREAM, SOL_SOCKET, socket
from struct import pack
from unittest import skipIf
from zope.interface.verify import verifyClass
from twisted.internet.interfaces import IPushProducer
from twisted.python.log import msg
from twisted.trial.unittest import TestCase
def test_noPendingTimerEvents(self):
    """
        Test reactor behavior (doIteration) when there are no pending time
        events.
        """
    ir = IOCPReactor()
    ir.wakeUp()
    self.assertFalse(ir.doIteration(None))