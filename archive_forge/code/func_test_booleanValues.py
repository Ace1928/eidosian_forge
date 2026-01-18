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
def test_booleanValues(self):
    """
        Verify that the Boolean parser parses 'True' and 'False', but nothing
        else.
        """
    b = amp.Boolean()
    self.assertTrue(b.fromString(b'True'))
    self.assertFalse(b.fromString(b'False'))
    self.assertRaises(TypeError, b.fromString, b'ninja')
    self.assertRaises(TypeError, b.fromString, b'true')
    self.assertRaises(TypeError, b.fromString, b'TRUE')
    self.assertEqual(b.toString(True), b'True')
    self.assertEqual(b.toString(False), b'False')