import base64
import cgi
import errno
import http.client
import os
import re
import socket
import ssl
import sys
import time
import urllib
import urllib.request
import weakref
from urllib.parse import urlencode, urljoin, urlparse
from ... import __version__ as breezy_version
from ... import config, debug, errors, osutils, trace, transport, ui, urlutils
from ...bzr.smart import medium
from ...trace import mutter, mutter_callsite
from ...transport import ConnectedTransport, NoSuchFile, UnusableRedirect
from . import default_user_agent, ssl
from .response import handle_response
def https_open(self, request):
    connection = request.connection
    if connection.sock is None and connection.proxied_host is not None and (request.get_method() != 'CONNECT'):
        connect = _ConnectRequest(request)
        response = self.parent.open(connect)
        if response.code != 200:
            raise errors.ConnectionError("Can't connect to {} via proxy {}".format(connect.proxied_host, self.host))
        connection.cleanup_pipe()
        connection.connect_to_origin()
        request.connection = connection
    return self.do_open(HTTPSConnection, request)