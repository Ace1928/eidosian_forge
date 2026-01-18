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
def test_empty_content(self):
    """'chunked
END
' is the complete encoding of a zero-length body.
        """
    decoder = protocol.ChunkedBodyDecoder()
    decoder.accept_bytes(b'chunked\n')
    decoder.accept_bytes(b'END\n')
    self.assertTrue(decoder.finished_reading)
    self.assertEqual(None, decoder.read_next_chunk())
    self.assertEqual(b'', decoder.unused_data)