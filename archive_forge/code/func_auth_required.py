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
def auth_required(self, request, headers):
    """Retry the request if the auth scheme is ours.

        :param request: The request needing authentication.
        :param headers: The headers for the authentication error response.
        :return: None or the response for the authenticated request.
        """
    if self._retry_count is None:
        self._retry_count = 1
    else:
        self._retry_count += 1
        if self._retry_count > self._max_retry:
            self._retry_count = None
            return None
    server_headers = headers.get_all(self.auth_required_header)
    if not server_headers:
        trace.mutter('%s not found', self.auth_required_header)
        return None
    auth = self.get_auth(request)
    auth['modified'] = False
    if auth.get('path', None) is None:
        parsed_url = urlutils.URL.from_string(request.get_full_url())
        self.update_auth(auth, 'protocol', parsed_url.scheme)
        self.update_auth(auth, 'host', parsed_url.host)
        self.update_auth(auth, 'port', parsed_url.port)
        self.update_auth(auth, 'path', parsed_url.path)
    for server_header in server_headers:
        matching_handler = self.auth_match(server_header, auth)
        if matching_handler:
            if request.get_header(self.auth_header, None) is not None and (not auth['modified']):
                return None
            best_scheme = auth.get('best_scheme', None)
            if best_scheme is None:
                best_scheme = auth['best_scheme'] = self.scheme
            if best_scheme != self.scheme:
                continue
            if self.requires_username and auth.get('user', None) is None:
                return None
            request.connection.cleanup_pipe()
            response = self.parent.open(request)
            if response:
                self.auth_successful(request, response)
            return response
    return None