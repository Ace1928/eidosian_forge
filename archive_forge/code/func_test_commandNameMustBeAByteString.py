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
def test_commandNameMustBeAByteString(self):
    """
        A L{Command} subclass cannot be defined with a C{commandName} that's
        not a byte string.
        """
    error = self.assertRaises(TypeError, type, 'NewCommand', (amp.Command,), {'commandName': 'FOO'})
    self.assertRegex(str(error), "^Command names must be byte strings, got: u?'FOO'$")