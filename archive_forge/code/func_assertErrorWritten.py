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
def assertErrorWritten(self, raised, reported):
    """
        Assert L{UnixApplicationRunner.postApplication} writes
        C{reported} to its status pipe if the service raises an
        exception whose message is C{raised}.
        """

    class FakeService(service.Service):

        def startService(self):
            raise RuntimeError(raised)
    errorService = FakeService()
    errorService.setServiceParent(self.runner.application)
    with AlternateReactor(FakeDaemonizingReactor()):
        self.assertRaises(RuntimeError, self.runner.postApplication)
    self.assertEqual(self.mockos.actions, [('chdir', '.'), ('umask', 63), ('fork', True), 'setsid', ('fork', True), ('write', -2, reported), ('unlink', 'twistd.pid')])
    self.assertEqual(self.mockos.closed, [-3, -2])