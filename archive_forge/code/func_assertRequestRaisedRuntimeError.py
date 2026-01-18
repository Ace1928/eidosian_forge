import os
import signal
import struct
import sys
from unittest import skipIf
from zope.interface import implementer
from twisted.internet import defer, error, protocol
from twisted.internet.address import IPv4Address
from twisted.internet.error import ProcessDone, ProcessTerminated
from twisted.python import components, failure
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.python.test.test_components import RegistryUsingMixin
from twisted.trial.unittest import TestCase
def assertRequestRaisedRuntimeError(self):
    """
        Assert that the request we just made raised a RuntimeError (and only a
        RuntimeError).
        """
    errors = self.flushLoggedErrors(RuntimeError)
    self.assertEqual(len(errors), 1, 'Multiple RuntimeErrors raised: %s' % '\n'.join([repr(error) for error in errors]))
    errors[0].trap(RuntimeError)