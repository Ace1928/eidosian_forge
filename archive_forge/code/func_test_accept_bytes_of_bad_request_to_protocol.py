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
def test_accept_bytes_of_bad_request_to_protocol(self):
    out_stream = BytesIO()
    smart_protocol = self.server_protocol_class(None, out_stream.write)
    smart_protocol.accept_bytes(b'abc')
    self.assertEqual(b'abc', smart_protocol.in_buffer)
    smart_protocol.accept_bytes(b'\n')
    self.assertEqual(self.response_marker + b"failed\nerror\x01Generic bzr smart protocol error: bad request 'abc'\n", out_stream.getvalue())
    self.assertTrue(smart_protocol._has_dispatched)
    self.assertEqual(0, smart_protocol.next_read_size())