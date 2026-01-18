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
def test_incomplete_length(self):
    """A chunk length hasn't been read until a newline byte has been read.
        """
    decoder = protocol.ChunkedBodyDecoder()
    decoder.accept_bytes(b'chunked\n')
    decoder.accept_bytes(b'9')
    self.assertEqual(1, decoder.next_read_size(), "The next_read_size hint should be 1, because we don't know the length yet.")
    decoder.accept_bytes(b'\n')
    self.assertEqual(9 + 4, decoder.next_read_size(), "The next_read_size hint should be the length of the chunk plus 4 (the length of the end-of-body marker: 'END\\n')")
    self.assertFalse(decoder.finished_reading)
    self.assertEqual(None, decoder.read_next_chunk())