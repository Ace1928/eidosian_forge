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
def bind_scheme_request(proxy, scheme):
    if proxy is None:
        return
    scheme_request = scheme + '_request'
    if self._debuglevel >= 3:
        print('Will bind {} for {!r}'.format(scheme_request, proxy))
    setattr(self, scheme_request, lambda request: self.set_proxy(request, scheme))