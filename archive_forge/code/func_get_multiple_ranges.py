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
def get_multiple_ranges(self, file, file_size, ranges):
    """Refuses the multiple ranges request"""
    tcs = self.server.test_case_server
    if tcs.range_limit is not None and len(ranges) > tcs.range_limit:
        file.close()
        self.send_error(400, 'Bad Request')
        return
    return http_server.TestingHTTPRequestHandler.get_multiple_ranges(self, file, file_size, ranges)