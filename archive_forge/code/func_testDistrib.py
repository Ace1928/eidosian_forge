from os.path import abspath
from xml.dom.minidom import parseString
from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.internet import defer, reactor
from twisted.logger import globalLogPublisher
from twisted.python import failure, filepath
from twisted.spread import pb
from twisted.spread.banana import SIZE_LIMIT
from twisted.test import proto_helpers
from twisted.trial.unittest import TestCase
from twisted.web import client, distrib, resource, server, static
from twisted.web.http_headers import Headers
from twisted.web.test._util import _render
from twisted.web.test.requesthelper import DummyChannel, DummyRequest
def testDistrib(self):
    r1 = resource.Resource()
    r1.putChild(b'there', static.Data(b'root', 'text/plain'))
    site1 = server.Site(r1)
    self.f1 = PBServerFactory(distrib.ResourcePublisher(site1))
    self.port1 = reactor.listenTCP(0, self.f1)
    self.sub = distrib.ResourceSubscription('127.0.0.1', self.port1.getHost().port)
    r2 = resource.Resource()
    r2.putChild(b'here', self.sub)
    f2 = MySite(r2)
    self.port2 = reactor.listenTCP(0, f2)
    agent = client.Agent(reactor)
    url = f'http://127.0.0.1:{self.port2.getHost().port}/here/there'
    url = url.encode('ascii')
    d = agent.request(b'GET', url)
    d.addCallback(client.readBody)
    d.addCallback(self.assertEqual, b'root')
    return d