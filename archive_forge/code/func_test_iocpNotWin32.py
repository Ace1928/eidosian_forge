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
@skipIf(platformType == 'win32', 'Test not applicable on Windows')
def test_iocpNotWin32(self):
    """
        --help-reactors should NOT display the iocp reactor on Windows
        """
    self.assertIn(twisted_reactors.iocp.description, self.message)
    self.assertIn('!' + twisted_reactors.iocp.shortName, self.message)