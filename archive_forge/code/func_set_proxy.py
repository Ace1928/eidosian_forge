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
def set_proxy(self, request, type):
    host = request.host
    if self.proxy_bypass(host):
        return request
    proxy = self.get_proxy_env_var(type)
    if self._debuglevel >= 3:
        print('set_proxy {}_request for {!r}'.format(type, proxy))
    parsed_url = transport.ConnectedTransport._split_url(proxy)
    if not parsed_url.host:
        raise urlutils.InvalidURL(proxy, 'No host component')
    if request.proxy_auth == {}:
        request.proxy_auth = {'host': parsed_url.host, 'port': parsed_url.port, 'user': parsed_url.user, 'password': parsed_url.password, 'protocol': parsed_url.scheme, 'path': None}
    if parsed_url.port is None:
        phost = parsed_url.host
    else:
        phost = parsed_url.host + ':%d' % parsed_url.port
    request.set_proxy(phost, type)
    if self._debuglevel >= 3:
        print('set_proxy: proxy set to {}://{}'.format(type, phost))
    return request