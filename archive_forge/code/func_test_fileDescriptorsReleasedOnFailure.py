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
def test_fileDescriptorsReleasedOnFailure(self):
    """
        L{_ExhaustsFileDescriptors.exhaust} closes any opened file
        descriptors if an exception occurs during its exhaustion loop.
        """
    fileDescriptors = []

    def failsAfterThree():
        if len(fileDescriptors) == 3:
            raise ValueError('test_fileDescriptorsReleasedOnFailure fake open exception')
        else:
            fd = os.dup(0)
            fileDescriptors.append(fd)
            return fd
    exhauster = _ExhaustsFileDescriptors(failsAfterThree)
    self.addCleanup(exhauster.release)
    self.assertRaises(ValueError, exhauster.exhaust)
    self.assertEqual(len(fileDescriptors), 3)
    self.assertEqual(exhauster.count(), 0)
    for fd in fileDescriptors:
        exception = self.assertRaises(OSError, os.fstat, fd)
        self.assertEqual(exception.errno, errno.EBADF)