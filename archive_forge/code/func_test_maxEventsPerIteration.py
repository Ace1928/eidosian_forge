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
def test_maxEventsPerIteration(self):
    """
        Verify that we don't lose an event when more than EVENTS_PER_LOOP
        events occur in the same reactor iteration
        """

    class FakeFD:
        counter = 0

        def logPrefix(self):
            return 'FakeFD'

        def cb(self, rc, bytes, evt):
            self.counter += 1
    ir = IOCPReactor()
    fd = FakeFD()
    event = _iocp.Event(fd.cb, fd)
    for _ in range(EVENTS_PER_LOOP + 1):
        ir.port.postEvent(0, KEY_NORMAL, event)
    ir.doIteration(None)
    self.assertEqual(fd.counter, EVENTS_PER_LOOP)
    ir.doIteration(0)
    self.assertEqual(fd.counter, EVENTS_PER_LOOP + 1)