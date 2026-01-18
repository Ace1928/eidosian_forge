import doctest
import errno
import os
import socket
import subprocess
import sys
import threading
import time
from io import BytesIO
from typing import Optional, Type
from testtools.matchers import DocTestMatches
import breezy
from ... import controldir, debug, errors, osutils, tests
from ... import transport as _mod_transport
from ... import urlutils
from ...tests import features, test_server
from ...transport import local, memory, remote, ssh
from ...transport.http import urllib
from .. import bzrdir
from ..remote import UnknownErrorFromSmartServer
from ..smart import client, medium, message, protocol
from ..smart import request as _mod_request
from ..smart import server as _mod_server
from ..smart import vfs
from . import test_smart
def test__send_read_response_sockets(self):
    listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen_sock.bind(('127.0.0.1', 0))
    listen_sock.listen(1)
    host, port = listen_sock.getsockname()
    client_medium = medium.SmartTCPClientMedium(host, port, '/')
    client_medium._ensure_connection()
    smart_client = client._SmartClient(client_medium)
    smart_request = client._SmartClientRequest(smart_client, b'hello', ())
    server_sock, _ = listen_sock.accept()
    server_sock.close()
    handler = smart_request._send(3)
    self.assertRaises(errors.ConnectionReset, handler.read_response_tuple, expect_body=False)