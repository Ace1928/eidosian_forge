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
def capture_connection(self, request, http_connection_class):
    """Capture or inject the request connection.

        Two cases:
        - the request have no connection: create a new one,

        - the request have a connection: this one have been used
          already, let's capture it, so that we can give it to
          another transport to be reused. We don't do that
          ourselves: the Transport object get the connection from
          a first request and then propagate it, from request to
          request or to cloned transports.
        """
    connection = request.connection
    if connection is None:
        connection = self.create_connection(request, http_connection_class)
        request.connection = connection
    connection.set_debuglevel(DEBUG)
    return request