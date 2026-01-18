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
def auth_match(self, header, auth):
    scheme, raw_auth = self._parse_auth_header(header)
    if scheme != self.scheme:
        return False
    req_auth = urllib.request.parse_keqv_list(urllib.request.parse_http_list(raw_auth))
    qop = req_auth.get('qop', None)
    if qop != 'auth':
        return False
    H, KD = get_digest_algorithm_impls(req_auth.get('algorithm', 'MD5'))
    if H is None:
        return False
    realm = req_auth.get('realm', None)
    self.update_auth(auth, 'scheme', scheme)
    self.update_auth(auth, 'realm', realm)
    if auth.get('user', None) is None or auth.get('password', None) is None:
        user, password = self.get_user_password(auth)
        self.update_auth(auth, 'user', user)
        self.update_auth(auth, 'password', password)
    try:
        if req_auth.get('algorithm', None) is not None:
            self.update_auth(auth, 'algorithm', req_auth.get('algorithm'))
        nonce = req_auth['nonce']
        if auth.get('nonce', None) != nonce:
            self.update_auth(auth, 'nonce_count', 0)
        self.update_auth(auth, 'nonce', nonce)
        self.update_auth(auth, 'qop', qop)
        auth['opaque'] = req_auth.get('opaque', None)
    except KeyError:
        return False
    return True