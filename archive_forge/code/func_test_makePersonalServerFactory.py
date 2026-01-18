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
def test_makePersonalServerFactory(self) -> None:
    """
        L{makePersonalServerFactory} returns a PB server factory which has
        as its root object a L{ResourcePublisher}.
        """
    site = Site(Data(b'foo bar', 'text/plain'))
    serverFactory = makePersonalServerFactory(site)
    self.assertIsInstance(serverFactory, PBServerFactory)
    self.assertIsInstance(serverFactory.root, ResourcePublisher)
    self.assertIdentical(serverFactory.root.site, site)