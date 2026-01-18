import errno
import socket
import socketserver
import threading
from breezy import osutils, tests
from breezy.tests import test_server
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_connection_timeout_suppressed(self):
    self.overrideAttr(test_server, '_DEFAULT_TESTING_CLIENT_TIMEOUT', 0.01)
    s = FakeServer()
    server_sock, client_sock = portable_socket_pair()
    test_server.TestingSmartConnectionHandler(server_sock, server_sock.getpeername(), s)