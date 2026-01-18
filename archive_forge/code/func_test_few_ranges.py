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
def test_few_ranges(self):
    t = self.get_readonly_transport()
    l = list(t.readv('a', ((0, 4), (1024, 4))))
    self.assertEqual(l[0], (0, b'0000'))
    self.assertEqual(l[1], (1024, b'0001'))
    self.assertEqual(1, self.get_readonly_server().GET_request_nb)