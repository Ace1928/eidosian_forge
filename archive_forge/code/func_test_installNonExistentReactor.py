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
def test_installNonExistentReactor(self):
    """
        Test that L{reactors.installReactor} raises L{reactors.NoSuchReactor}
        when asked to install a reactor which it cannot find.
        """
    self.pluginResults = []
    self.assertRaises(reactors.NoSuchReactor, reactors.installReactor, 'somereactor')