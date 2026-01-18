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
def test_evaluate_proxy_bypass_unknown(self):
    """The host is not explicitly proxied"""
    self.assertEvaluateProxyBypass(None, 'example.com', 'not.example.com')
    self.assertEvaluateProxyBypass(None, 'bzr.example.com', 'example.com')