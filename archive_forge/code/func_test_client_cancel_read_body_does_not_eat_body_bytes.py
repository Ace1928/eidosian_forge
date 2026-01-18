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
def test_client_cancel_read_body_does_not_eat_body_bytes(self):
    server_bytes = self.response_marker + b'success\nok\n7\n1234567done\n'
    input = BytesIO(server_bytes)
    output = BytesIO()
    client_medium = medium.SmartSimplePipesClientMedium(input, output, 'base')
    request = client_medium.get_request()
    smart_protocol = self.client_protocol_class(request)
    smart_protocol.call(b'foo')
    smart_protocol.read_response_tuple(True)
    smart_protocol.cancel_read_body()
    self.assertEqual(len(self.response_marker + b'success\nok\n'), input.tell())
    self.assertRaises(errors.ReadingCompleted, smart_protocol.read_body_bytes)