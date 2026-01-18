import os
import socket
import sys
import time
from breezy import config, controldir, errors, tests
from breezy import transport as _mod_transport
from breezy import ui
from breezy.osutils import lexists
from breezy.tests import TestCase, TestCaseWithTransport, TestSkipped, features
from breezy.tests.http_server import HttpServer
def test__get_requests(self):
    self.checkGetRequests([(0, 100)], [(0, 20), (30, 50), (20, 10), (80, 20)])
    self.checkGetRequests([(0, 20), (30, 50)], [(10, 10), (30, 20), (0, 10), (50, 30)])
    self.checkGetRequests([(0, 32768), (32768, 32768), (65536, 464)], [(0, 40000), (40000, 100), (40100, 1900), (42000, 24000)])