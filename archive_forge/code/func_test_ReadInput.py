import json
import os
import sys
from io import BytesIO
from twisted.internet import address, error, interfaces, reactor
from twisted.internet.error import ConnectionLost
from twisted.python import failure, log, util
from twisted.trial import unittest
from twisted.web import client, http, http_headers, resource, server, twcgi
from twisted.web.http import INTERNAL_SERVER_ERROR, NOT_FOUND
from twisted.web.test._util import _render
from twisted.web.test.requesthelper import DummyChannel, DummyRequest
import os, sys
import sys
import json
import os
import os
def test_ReadInput(self):
    cgiFilename = os.path.abspath(self.mktemp())
    with open(cgiFilename, 'wt') as cgiFile:
        cgiFile.write(READINPUT_CGI)
    portnum = self.startServer(cgiFilename)
    agent = client.Agent(reactor)
    url = 'http://localhost:%d/cgi' % (portnum,)
    url = url.encode('ascii')
    d = agent.request(uri=url, method=b'POST', bodyProducer=client.FileBodyProducer(BytesIO(b'Here is your stdin')))
    d.addCallback(client.readBody)
    d.addCallback(self._test_ReadInput_1)
    return d