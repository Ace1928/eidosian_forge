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
def getLinkLocalIPv6Address():
    """
    Find and return a configured link local IPv6 address including a scope
    identifier using the % separation syntax.  If the system has no link local
    IPv6 addresses, raise L{SkipTest} instead.

    @raise SkipTest: if no link local address can be found or if the
        C{netifaces} module is not available.

    @return: a C{str} giving the address
    """
    addresses = getLinkLocalIPv6Addresses()
    if addresses:
        return addresses[0]
    raise SkipTest('Link local IPv6 address unavailable')