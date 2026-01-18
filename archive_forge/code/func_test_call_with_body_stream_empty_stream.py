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
def test_call_with_body_stream_empty_stream(self):
    """call_with_body_stream with an empty stream."""
    requester, output = self.make_client_encoder_and_output()
    requester.set_headers({})
    stream = []
    requester.call_with_body_stream((b'one arg',), stream)
    self.assertEqual(b'bzr message 3 (bzr 1.6)\n\x00\x00\x00\x02des\x00\x00\x00\x0bl7:one argee', output.getvalue())