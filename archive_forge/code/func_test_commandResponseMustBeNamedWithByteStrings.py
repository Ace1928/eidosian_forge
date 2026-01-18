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
def test_commandResponseMustBeNamedWithByteStrings(self):
    """
        A L{Command} subclass's C{response} must have byte string names.
        """
    error = self.assertRaises(TypeError, type, 'NewCommand', (amp.Command,), {'response': [('foo', None)]})
    self.assertRegex(str(error), "^Response names must be byte strings, got: u?'foo'$")