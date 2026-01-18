import os
import hamcrest
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet.error import ConnectionDone
from twisted.internet.testing import waitUntilAllDisconnected
from twisted.protocols import basic
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.test.test_tcp import ProperlyCloseFilesMixin
from twisted.trial.unittest import TestCase
from zope.interface import implementer
def setupServerAndClient(self, clientArgs, clientKwArgs, serverArgs, serverKwArgs):
    self.clientBase, self.clientCtxFactory = self.makeContextFactory(*clientArgs, **clientKwArgs)
    self.serverBase, self.serverCtxFactory = self.makeContextFactory(*serverArgs, **serverKwArgs)