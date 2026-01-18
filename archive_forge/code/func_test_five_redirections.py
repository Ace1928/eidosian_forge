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
def test_five_redirections(self):
    t = self.get_old_transport()
    old_prefix = 'http://{}:{}'.format(self.old_server.host, self.old_server.port)
    new_prefix = 'http://{}:{}'.format(self.new_server.host, self.new_server.port)
    self.old_server.redirections = [('/1(.*)', '%s/2\\1' % old_prefix, 302), ('/2(.*)', '%s/3\\1' % old_prefix, 303), ('/3(.*)', '%s/4\\1' % old_prefix, 307), ('/4(.*)', '%s/5\\1' % new_prefix, 301), ('(/[^/]+)', '%s/1\\1' % old_prefix, 301)]
    self.assertEqual(b'redirected 5 times', t.request('GET', t._remote_path('a'), retries=6).read())