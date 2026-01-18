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
def retry_or_raise(self, http_class, request, first_try):
    """Retry the request (once) or raise the exception.

        urllib.request raises exception of application level kind, we
        just have to translate them.

        http.client can raise exceptions of transport level (badly
        formatted dialog, loss of connexion or socket level
        problems). In that case we should issue the request again
        (http.client will close and reopen a new connection if
        needed).
        """
    exc_type, exc_val, exc_tb = sys.exc_info()
    if exc_type == socket.gaierror:
        origin_req_host = request.origin_req_host
        raise errors.ConnectionError("Couldn't resolve host '%s'" % origin_req_host, orig_error=exc_val)
    elif isinstance(exc_val, http.client.ImproperConnectionState):
        raise exc_val.with_traceback(exc_tb)
    elif first_try:
        if self._debuglevel >= 2:
            print('Received exception: [%r]' % exc_val)
            print('  On connection: [%r]' % request.connection)
            method = request.get_method()
            url = request.get_full_url()
            print('  Will retry, {} {!r}'.format(method, url))
        request.connection.close()
        response = self.do_open(http_class, request, False)
    else:
        if self._debuglevel >= 2:
            print('Received second exception: [%r]' % exc_val)
            print('  On connection: [%r]' % request.connection)
        if exc_type in (http.client.BadStatusLine, http.client.UnknownProtocol):
            my_exception = errors.InvalidHttpResponse(request.get_full_url(), 'Bad status line received', orig_error=exc_val)
        elif isinstance(exc_val, socket.error) and len(exc_val.args) and (exc_val.args[0] in (errno.ECONNRESET, 10053, 10054)):
            raise errors.ConnectionReset('Connection lost while sending request.')
        else:
            selector = request.selector
            my_exception = errors.ConnectionError(msg='while sending {} {}:'.format(request.get_method(), selector), orig_error=exc_val)
        if self._debuglevel >= 2:
            print('On connection: [%r]' % request.connection)
            method = request.get_method()
            url = request.get_full_url()
            print('  Failed again, {} {!r}'.format(method, url))
            print('  Will raise: [%r]' % my_exception)
        raise my_exception.with_traceback(exc_tb)
    return response