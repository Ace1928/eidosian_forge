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
def test_add_header_resource(self) -> None:
    """
        When --add-header is specified, the resource is a composition that adds
        headers.
        """
    options = Options()
    options.parseOptions(['--add-header', 'K1: V1', '--add-header', 'K2: V2'])
    service = makeService(options)
    resource = service.services[0].factory.resource
    self.assertIsInstance(resource, _AddHeadersResource)
    self.assertEqual(resource._headers, [('K1', 'V1'), ('K2', 'V2')])
    self.assertIsInstance(resource._originalResource, demo.Test)