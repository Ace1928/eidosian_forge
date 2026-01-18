import errno
import socket
import socketserver
import threading
from breezy import osutils, tests
from breezy.tests import test_server
from breezy.tests.scenarios import load_tests_apply_scenarios
def assertClientAddr(self, client, server, conn_rank):
    conn = self.get_server_connection(server, conn_rank)
    self.assertEqual(client.sock.getsockname(), conn[1])