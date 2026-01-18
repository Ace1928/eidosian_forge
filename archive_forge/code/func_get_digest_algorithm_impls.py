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
def get_digest_algorithm_impls(algorithm):
    H = None
    KD = None
    if algorithm == 'MD5':

        def H(x):
            return osutils.md5(x).hexdigest()
    elif algorithm == 'SHA':
        H = osutils.sha_string
    if H is not None:

        def KD(secret, data):
            return H('{}:{}'.format(secret, data).encode('utf-8'))
    return (H, KD)