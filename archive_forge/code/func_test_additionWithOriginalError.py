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
def test_additionWithOriginalError(self):
    """
        Verify that errors specified in a command's superclass are respected
        even if that command defines new errors itself.
        """
    return self.errorCheck(InheritedError, AddedCommandProtocol, AddErrorsCommand, other=False)