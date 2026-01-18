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
def test_noDuplicateContentTypeHeaders(self):
    """
        If the CGI script emits a I{content-type} header, make sure that the
        server doesn't add an additional (duplicate) one, as per ticket 4786.
        """
    cgiFilename = self.writeCGI(NO_DUPLICATE_CONTENT_TYPE_HEADER_CGI)
    portnum = self.startServer(cgiFilename)
    url = 'http://localhost:%d/cgi' % (portnum,)
    url = url.encode('ascii')
    agent = client.Agent(reactor)
    d = agent.request(b'GET', url)
    d.addCallback(discardBody)

    def checkResponse(response):
        self.assertEqual(response.headers.getRawHeaders('content-type'), ['text/cgi-duplicate-test'])
        return response
    d.addCallback(checkResponse)
    return d