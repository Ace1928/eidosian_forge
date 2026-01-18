import datetime
from io import BytesIO, StringIO
from unittest import skipIf
from twisted.internet import defer, reactor
from twisted.internet.error import ConnectionDone
from twisted.internet.testing import EventLoggingObserver, MemoryReactor
from twisted.logger import (
from twisted.python import failure
from twisted.python.compat import nativeString, networkString
from twisted.python.reflect import namedModule
from twisted.trial import unittest
from twisted.web import client, http, server, static, xmlrpc
from twisted.web.test.test_web import DummyRequest
from twisted.web.xmlrpc import (
def check_client_headers(ign):
    self.assertEqual(self.factories[0].sent_headers[b'user-agent'], b'Twisted/XMLRPClib')
    self.assertEqual(self.factories[0].sent_headers[b'content-type'], b'text/xml; charset=utf-8')
    self.assertEqual(self.factories[0].sent_headers[b'content-length'], b'155')