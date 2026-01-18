import errno
import socket
import socketserver
import threading
from breezy import osutils, tests
from breezy.tests import test_server
from breezy.tests.scenarios import load_tests_apply_scenarios
def handle_connection(request):
    request.readline()
    self.connection_thread = threading.currentThread()
    self.connection_thread.set_sync_event(caught)
    raise FailToRespond()