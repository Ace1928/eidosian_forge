import copy
import os
import pickle
from io import StringIO
from unittest import skipIf
from twisted.application import app, internet, reactors, service
from twisted.application.internet import backoffPolicy
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet.testing import MemoryReactor
from twisted.persisted import sob
from twisted.plugins import twisted_reactors
from twisted.protocols import basic, wire
from twisted.python import usage
from twisted.python.runtime import platformType
from twisted.python.test.modules_helpers import TwistedModulesMixin
from twisted.trial.unittest import SkipTest, TestCase
def test_calculates_correct_values(self):
    """
        Test that L{backoffPolicy()} calculates expected values
        """
    pol = backoffPolicy(1.0, 60.0, 1.5, jitter=lambda: 1)
    self.assertAlmostEqual(pol(0), 2)
    self.assertAlmostEqual(pol(1), 2.5)
    self.assertAlmostEqual(pol(10), 58.6650390625)
    self.assertEqual(pol(20), 61)
    self.assertEqual(pol(100), 61)