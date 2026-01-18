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
def test_callRemoteStringRequiresAnswerFalse(self):
    """
        L{BoxDispatcher.callRemoteString} returns L{None} if C{requiresAnswer}
        is C{False}.
        """
    c, s, p = connectedServerAndClient()
    ret = c.callRemoteString(b'WTF', requiresAnswer=False)
    self.assertIsNone(ret)