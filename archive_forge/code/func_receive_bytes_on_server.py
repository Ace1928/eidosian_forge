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
def receive_bytes_on_server(self, sock, bytes):
    """Accept a connection on sock and read 3 bytes.

        The bytes are appended to the list bytes.

        :return: a Thread which is running to do the accept and recv.
        """

    def _receive_bytes_on_server():
        connection, address = sock.accept()
        bytes.append(osutils.recv_all(connection, 3))
        connection.close()
    t = threading.Thread(target=_receive_bytes_on_server)
    t.start()
    return t