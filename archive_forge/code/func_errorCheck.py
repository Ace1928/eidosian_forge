import datetime
import decimal
from typing import ClassVar, Dict, Type, TypeVar
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.internet import address, defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.protocols import amp
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.test import iosim
from twisted.trial.unittest import TestCase
def errorCheck(self, err, proto, cmd, **kw):
    """
        Check that the appropriate kind of error is raised when a given command
        is sent to a given protocol.
        """
    c, s, p = connectedServerAndClient(ServerClass=proto, ClientClass=proto)
    d = c.callRemote(cmd, **kw)
    d2 = self.failUnlessFailure(d, err)
    p.flush()
    return d2