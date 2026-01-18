import errno
import gc
import io
import os
import socket
from functools import wraps
from typing import Callable, ClassVar, List, Mapping, Optional, Sequence, Type
from unittest import skipIf
from zope.interface import Interface, implementer
from zope.interface.verify import verifyClass, verifyObject
import attr
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import (
from twisted.internet.endpoints import TCP4ClientEndpoint, TCP4ServerEndpoint
from twisted.internet.error import (
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.internet.tcp import (
from twisted.internet.test.connectionmixins import (
from twisted.internet.test.reactormixins import (
from twisted.internet.testing import MemoryReactor, StringTransport
from twisted.logger import Logger
from twisted.python import log
from twisted.python.failure import Failure
from twisted.python.runtime import platform
from twisted.test.test_tcp import (
from twisted.trial.unittest import SkipTest, SynchronousTestCase, TestCase
def test_flushesOnExitWithException(self):
    """
        The context manager flushes its buffered logs when the block
        terminates because of an exception.
        """

    class TestException(Exception):
        """
            An exception only raised by this test.
            """
    with self.assertRaises(TestException):
        with self.logBuffer as logger:
            logger.info('An event')
            self.assertFalse(self.events)
            raise TestException()
    self.assertEqual(1, len(self.events))
    [event] = self.events
    self.assertEqual(event['log_format'], 'An event')
    self.assertEqual(event['log_namespace'], self.namespace)