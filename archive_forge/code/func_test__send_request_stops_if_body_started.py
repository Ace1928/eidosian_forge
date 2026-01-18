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
def test__send_request_stops_if_body_started(self):
    from io import BytesIO
    response = BytesIO()

    class FailAfterFirstWrite(BytesIO):
        """Allow one 'write' call to pass, fail the rest"""

        def __init__(self):
            BytesIO.__init__(self)
            self._first = True

        def write(self, s):
            if self._first:
                self._first = False
                return BytesIO.write(self, s)
            raise OSError(errno.EINVAL, 'invalid file handle')
    output = FailAfterFirstWrite()
    vendor = FirstRejectedBytesIOSSHVendor(response, output, fail_at_write=False)
    ssh_params = medium.SSHParams('a host', 'a port', 'a user', 'a pass')
    client_medium = medium.SmartSSHClientMedium('base', ssh_params, vendor)
    smart_client = client._SmartClient(client_medium, headers={})
    smart_request = client._SmartClientRequest(smart_client, b'hello', (), body_stream=[b'a', b'b'])
    self.assertRaises(errors.ConnectionReset, smart_request._send, 3)
    self.assertEqual([('connect_ssh', 'a user', 'a pass', 'a host', 'a port', ['bzr', 'serve', '--inet', '--directory=/', '--allow-writes']), ('close',)], vendor.calls)
    self.assertEqual(b'bzr message 3 (bzr 1.6)\n\x00\x00\x00\x02des\x00\x00\x00\tl5:helloe', output.getvalue())