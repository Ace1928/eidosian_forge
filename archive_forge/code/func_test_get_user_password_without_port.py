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
def test_get_user_password_without_port(self):
    """We cope if urllib2 doesn't tell us the port.

        See https://bugs.launchpad.net/bzr/+bug/654684
        """
    user = 'joe'
    password = 'foo'
    _setup_authentication_config(scheme='http', host='localhost', user=user, password=password)
    handler = HTTPAuthHandler()
    got_pass = handler.get_user_password(dict(user='joe', protocol='http', host='localhost', path='/', realm='Realm'))
    self.assertEqual((user, password), got_pass)