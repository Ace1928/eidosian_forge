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
def test_accept_excess_bytes_after_body(self):
    server_protocol = self.build_protocol_waiting_for_body()
    server_protocol.accept_bytes(b'7\nabcdefgdone\n' + self.response_marker)
    self.assertTrue(self.end_received)
    self.assertEqual(self.response_marker, server_protocol.unused_data)
    self.assertEqual(b'', server_protocol.in_buffer)
    server_protocol.accept_bytes(b'Y')
    self.assertEqual(self.response_marker + b'Y', server_protocol.unused_data)
    self.assertEqual(b'', server_protocol.in_buffer)