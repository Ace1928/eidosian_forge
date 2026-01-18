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
def test_multidigit_length(self):
    """Lengths in the chunk prefixes can have multiple digits."""
    decoder = protocol.ChunkedBodyDecoder()
    decoder.accept_bytes(b'chunked\n')
    length = 291
    chunk_prefix = hex(length).encode('ascii') + b'\n'
    chunk_bytes = b'z' * length
    finish = b'END\n'
    decoder.accept_bytes(chunk_prefix + chunk_bytes + finish)
    self.assertTrue(decoder.finished_reading)
    self.assertEqual(chunk_bytes, decoder.read_next_chunk())