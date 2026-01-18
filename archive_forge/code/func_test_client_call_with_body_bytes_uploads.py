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
def test_client_call_with_body_bytes_uploads(self):
    expected_bytes = self.request_marker + b'foo\n7\nabcdefgdone\n'
    input = BytesIO(b'\n')
    output = BytesIO()
    client_medium = medium.SmartSimplePipesClientMedium(input, output, 'base')
    request = client_medium.get_request()
    smart_protocol = self.client_protocol_class(request)
    smart_protocol.call_with_body_bytes((b'foo',), b'abcdefg')
    self.assertEqual(expected_bytes, output.getvalue())