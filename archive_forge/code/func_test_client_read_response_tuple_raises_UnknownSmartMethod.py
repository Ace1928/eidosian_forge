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
def test_client_read_response_tuple_raises_UnknownSmartMethod(self):
    """read_response_tuple raises UnknownSmartMethod if the response says
        the server did not recognise the request.
        """
    server_bytes = protocol.RESPONSE_VERSION_TWO + b'failed\n' + b"error\x01Generic bzr smart protocol error: bad request 'foo'\n"
    input = BytesIO(server_bytes)
    output = BytesIO()
    client_medium = medium.SmartSimplePipesClientMedium(input, output, 'base')
    request = client_medium.get_request()
    smart_protocol = protocol.SmartClientRequestProtocolTwo(request)
    smart_protocol.call(b'foo')
    self.assertRaises(errors.UnknownSmartMethod, smart_protocol.read_response_tuple)
    self.assertRaises(errors.ReadingCompleted, smart_protocol.read_body_bytes)