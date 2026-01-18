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
def test_inheritedErrorAddition(self):
    """
        Verify that new errors specified in a subclass of an existing command
        are honored even if the superclass defines some errors.
        """
    return self.errorCheck(OtherInheritedError, AddedCommandProtocol, AddErrorsCommand, other=True)