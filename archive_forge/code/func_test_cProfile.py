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
@skipIf(not cProfile, 'cProfile module not available')
def test_cProfile(self):
    """
        L{app.CProfileRunner.run} should call the C{run} method of the
        reactor and save profile data in the specified file.
        """
    config = twistd.ServerOptions()
    config['profile'] = self.mktemp()
    config['profiler'] = 'cProfile'
    profiler = app.AppProfiler(config)
    reactor = DummyReactor()
    profiler.run(reactor)
    self.assertTrue(reactor.called)
    with open(config['profile']) as f:
        data = f.read()
    self.assertIn('run', data)
    self.assertIn('function calls', data)