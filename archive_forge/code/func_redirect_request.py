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
def redirect_request(self, req, fp, code, msg, headers, newurl):
    """See urllib.request.HTTPRedirectHandler.redirect_request"""
    origin_req_host = req.origin_req_host
    if code in (301, 302, 303, 307, 308):
        return Request(req.get_method(), newurl, headers=req.headers, origin_req_host=origin_req_host, unverifiable=True, connection=None, parent=req)
    else:
        raise urllib.request.HTTPError(req.get_full_url(), code, msg, headers, fp)