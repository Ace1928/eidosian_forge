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
@skipIf(not _twistd_unix, 'twistd unix not available')
def test_getLogObserverSyslog(self):
    """
        If C{syslog} is set to C{True}, L{UnixAppLogger._getLogObserver} starts
        a L{syslog.SyslogObserver} with given C{prefix}.
        """
    logs = _setupSyslog(self)
    logger = UnixAppLogger({'syslog': True, 'prefix': 'test-prefix'})
    observer = logger._getLogObserver()
    self.assertEqual(logs, ['test-prefix'])
    observer({'a': 'b'})
    self.assertEqual(logs, ['test-prefix', {'a': 'b'}])