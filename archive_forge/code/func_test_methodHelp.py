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
def test_methodHelp(self):
    inputOutputs = [('defer', 'Help for defer.'), ('fail', ''), ('dict', 'Help for dict.')]
    dl = []
    for meth, expected in inputOutputs:
        d = self.proxy().callRemote('system.methodHelp', meth)
        d.addCallback(self.assertEqual, expected)
        dl.append(d)
    return defer.DeferredList(dl, fireOnOneErrback=True)