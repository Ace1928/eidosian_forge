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
def test_invalidDatetime(self):
    """
        L{amp.DateTime.toString} raises L{ValueError} when passed a naive
        datetime (a datetime with no timezone information).
        """
    d = amp.DateTime()
    self.assertRaises(ValueError, d.toString, datetime.datetime(2010, 12, 25, 0, 0, 0))