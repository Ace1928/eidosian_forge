import errno
import socket
import socketserver
import threading
from breezy import osutils, tests
from breezy.tests import test_server
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_handle_request_closes_if_it_doesnt_process(self):
    server = self.get_server()
    client = self.get_client()
    server.server.serving = False
    try:
        client.connect((server.host, server.port))
        self.assertEqual(b'', client.read())
    except OSError as e:
        if e.errno != errno.ECONNRESET:
            raise