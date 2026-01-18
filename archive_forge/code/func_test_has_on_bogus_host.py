import io
import socket
import sys
import threading
from http.client import UnknownProtocol, parse_headers
from http.server import SimpleHTTPRequestHandler
import breezy
from .. import (config, controldir, debug, errors, osutils, tests, trace,
from ..bzr import remote as _mod_remote
from ..transport import remote
from ..transport.http import urllib
from ..transport.http.urllib import (AbstractAuthHandler, BasicAuthHandler,
from . import features, http_server, http_utils, test_server
from .scenarios import load_tests_apply_scenarios, multiply_scenarios
def test_has_on_bogus_host(self):
    default_timeout = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(2)
        s = socket.socket()
        s.bind(('localhost', 0))
        t = self._transport('http://%s:%s/' % s.getsockname())
        self.assertRaises(errors.ConnectionError, t.has, 'foo/bar')
    finally:
        socket.setdefaulttimeout(default_timeout)