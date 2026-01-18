from contextlib import closing, contextmanager
import errno
import socket
import threading
import time
import http.client
import pytest
import cheroot.server
from cheroot.test import webtest
import cheroot.wsgi
def get_server_client(server):
    """Create and return a test client for the given server."""
    return _TestClient(server)