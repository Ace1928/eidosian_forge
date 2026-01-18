import base64
import xmlrpc.client as xmlrpclib
from urllib.parse import urlparse
from xmlrpc.client import Binary, Boolean, DateTime, Fault
from twisted.internet import defer, error, protocol
from twisted.logger import Logger
from twisted.python import failure, reflect
from twisted.python.compat import nativeString
from twisted.web import http, resource, server
def render_POST(self, request):
    request.content.seek(0, 0)
    request.setHeader(b'content-type', b'text/xml; charset=utf-8')
    try:
        args, functionPath = xmlrpclib.loads(request.content.read(), use_datetime=self.useDateTime)
    except Exception as e:
        f = Fault(self.FAILURE, f"Can't deserialize input: {e}")
        self._cbRender(f, request)
    else:
        try:
            function = self.lookupProcedure(functionPath)
        except Fault as f:
            self._cbRender(f, request)
        else:
            responseFailed = []
            request.notifyFinish().addErrback(responseFailed.append)
            if getattr(function, 'withRequest', False):
                d = defer.maybeDeferred(function, request, *args)
            else:
                d = defer.maybeDeferred(function, *args)
            d.addErrback(self._ebRender)
            d.addCallback(self._cbRender, request, responseFailed)
    return server.NOT_DONE_YET