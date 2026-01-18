import errno
import inspect
import os
import pickle
import signal
import sys
from io import StringIO
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted import internet, logger, plugin
from twisted.application import app, reactors, service
from twisted.application.service import IServiceMaker
from twisted.internet.base import ReactorBase
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IReactorDaemonize, _ISupportsExitSignalCapturing
from twisted.internet.test.modulehelpers import AlternateReactor
from twisted.internet.testing import MemoryReactor
from twisted.logger import ILogObserver, globalLogBeginner, globalLogPublisher
from twisted.python import util
from twisted.python.components import Componentized
from twisted.python.fakepwd import UserDatabase
from twisted.python.log import ILogObserver as LegacyILogObserver, textFromEventDict
from twisted.python.reflect import requireModule
from twisted.python.runtime import platformType
from twisted.python.usage import UsageError
from twisted.scripts import twistd
from twisted.test.test_process import MockOS
from twisted.trial.unittest import TestCase
def test_exitWithSignal(self):
    """
        exitWithSignal replaces the existing signal handler with the default
        handler and sends the replaced signal to the current process.
        """
    fakeSignalArgs = [None, None]

    def fake_signal(sig, handler):
        fakeSignalArgs[0] = sig
        fakeSignalArgs[1] = handler
    self.patch(signal, 'signal', fake_signal)
    app._exitWithSignal(signal.SIGINT)
    self.assertEquals(fakeSignalArgs[0], signal.SIGINT)
    self.assertEquals(fakeSignalArgs[1], signal.SIG_DFL)
    self.assertEquals(self.fakeKillArgs[0], os.getpid())
    self.assertEquals(self.fakeKillArgs[1], signal.SIGINT)