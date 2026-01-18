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
@skipIf(not IReactorUNIX.providedBy(reactor), 'The reactor does not support UNIX domain sockets')
def test_defaultPersonalPath(self) -> None:
    """
        If the I{--port} option not specified but the I{--personal} option is,
        L{Options} defaults the port to C{UserDirectory.userSocketName} in the
        user's home directory.
        """
    options = Options()
    options.parseOptions(['--personal'])
    path = os.path.expanduser(os.path.join('~', UserDirectory.userSocketName))
    self.assertEqual(options['ports'][0], f'unix:{path}')