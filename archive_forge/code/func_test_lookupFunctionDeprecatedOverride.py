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
def test_lookupFunctionDeprecatedOverride(self):
    """
        Subclasses which override locateResponder under its old name,
        lookupFunction, should have the override invoked instead.  (This tests
        an AMP subclass, because in the version of the code that could invoke
        this deprecated code path, there was no L{CommandLocator}.)
        """
    locator = OverrideLocatorAMP()
    customResponderObject = self.assertWarns(PendingDeprecationWarning, 'Override locateResponder, not lookupFunction.', __file__, lambda: locator.locateResponder(b'custom'))
    self.assertEqual(locator.customResponder, customResponderObject)
    normalResponderObject = self.assertWarns(PendingDeprecationWarning, 'Override locateResponder, not lookupFunction.', __file__, lambda: locator.locateResponder(b'simple'))
    result = normalResponderObject(amp.Box(greeting=b'ni hao', cookie=b'5'))

    def done(values):
        self.assertEqual(values, amp.AmpBox(cookieplus=b'8'))
    return result.addCallback(done)