import errno
import socket
import socketserver
import threading
from breezy import osutils, tests
from breezy.tests import test_server
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_exception_swallowed_while_serving(self):
    caught = threading.Event()
    caught.clear()
    self.connection_thread = None

    class CantServe(Exception):
        pass

    class FailingWhileServingConnectionHandler(TCPConnectionHandler):

        def handle(request):
            self.connection_thread = threading.currentThread()
            self.connection_thread.set_sync_event(caught)
            raise CantServe()
    server = self.get_server(connection_handler_class=FailingWhileServingConnectionHandler)
    self.assertEqual(True, server.server.serving)
    server.set_ignored_exceptions(CantServe)
    client = self.get_client()
    client.connect((server.host, server.port))
    caught.wait()
    self.assertEqual(b'', client.read())
    self.assertIs(None, self.connection_thread.pending_exception())
    self.assertIs(None, server.pending_exception())