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
def test_abspath_root_sibling_server(self):
    server = stub_sftp.SFTPSiblingAbsoluteServer()
    server.start_server()
    self.addCleanup(server.stop_server)
    transport = _mod_transport.get_transport_from_url(server.get_url())
    self.assertFalse(transport.abspath('/').endswith('/~/'))
    self.assertTrue(transport.abspath('/').endswith('/'))
    del transport