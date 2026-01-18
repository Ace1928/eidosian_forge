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
def assertServerToClientEncoding(self, expected_bytes, expected_tuple, input_tuples):
    """Assert that each input_tuple serialises as expected_bytes, and the
        bytes deserialise as expected_tuple.
        """
    for input_tuple in input_tuples:
        server_protocol, server_output = self.make_server_protocol()
        server_protocol._send_response(_mod_request.SuccessfulSmartServerResponse(input_tuple))
        self.assertEqual(expected_bytes, server_output.getvalue())
    requester, response_handler = self.make_client_protocol(expected_bytes)
    requester.call(b'foo')
    self.assertEqual(expected_tuple, response_handler.read_response_tuple())