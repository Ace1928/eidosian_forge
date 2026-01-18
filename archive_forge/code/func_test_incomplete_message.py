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
def test_incomplete_message(self):
    """A decoder will keep signalling that it needs more bytes via
        next_read_size() != 0 until it has seen a complete message, regardless
        which state it is in.
        """
    headers = b'\x00\x00\x00\x02de'
    response_status = b'oS'
    args = b's\x00\x00\x00\x02le'
    body = b'b\x00\x00\x00\x04BODY'
    end = b'e'
    simple_response = headers + response_status + args + body + end
    decoder, response_handler = self.make_logging_response_decoder()
    for byte in bytearray(simple_response):
        self.assertNotEqual(0, decoder.next_read_size())
        decoder.accept_bytes(bytes([byte]))
    self.assertEqual(0, decoder.next_read_size())