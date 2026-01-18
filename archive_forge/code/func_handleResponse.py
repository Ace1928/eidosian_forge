import base64
import xmlrpc.client as xmlrpclib
from urllib.parse import urlparse
from xmlrpc.client import Binary, Boolean, DateTime, Fault
from twisted.internet import defer, error, protocol
from twisted.logger import Logger
from twisted.python import failure, reflect
from twisted.python.compat import nativeString
from twisted.web import http, resource, server
def handleResponse(self, contents):
    """
        Handle the XML-RPC response received from the server.

        Specifically, disconnect from the server and store the XML-RPC
        response so that it can be properly handled when the disconnect is
        finished.
        """
    self.transport.loseConnection()
    self._response = contents