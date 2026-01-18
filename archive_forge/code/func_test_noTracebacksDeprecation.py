from __future__ import annotations
import os
import stat
from typing import cast
from unittest import skipIf
from twisted.internet import endpoints, reactor
from twisted.internet.interfaces import IReactorCore, IReactorUNIX
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.python.threadpool import ThreadPool
from twisted.python.usage import UsageError
from twisted.spread.pb import PBServerFactory
from twisted.trial.unittest import TestCase
from twisted.web import demo
from twisted.web.distrib import ResourcePublisher, UserDirectory
from twisted.web.script import PythonScript
from twisted.web.server import Site
from twisted.web.static import Data, File
from twisted.web.tap import (
from twisted.web.test.requesthelper import DummyRequest
from twisted.web.twcgi import CGIScript
from twisted.web.wsgi import WSGIResource
def test_noTracebacksDeprecation(self) -> None:
    """
        Passing --notracebacks is deprecated.
        """
    options = Options()
    options.parseOptions(['--notracebacks'])
    makeService(options)
    warnings = self.flushWarnings([self.test_noTracebacksDeprecation])
    self.assertEqual(warnings[0]['category'], DeprecationWarning)
    self.assertEqual(warnings[0]['message'], '--notracebacks was deprecated in Twisted 19.7.0')
    self.assertEqual(len(warnings), 1)