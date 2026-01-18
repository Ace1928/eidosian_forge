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
def test_first_response_is_error(self):
    """If the server replies with an error, then the version detection
        should be complete.

        This test is very similar to test_version_two_server, but catches a bug
        we had in the case where the first reply was an error response.
        """
    medium = MockMedium()
    smart_client = client._SmartClient(medium, headers={})
    message_start = protocol.MESSAGE_VERSION_THREE + b'\x00\x00\x00\x02de'
    medium.expect_request(message_start + b's\x00\x00\x00\x10l11:method-nameee', b'bzr response 2\nfailed\n\n')
    medium.expect_disconnect()
    medium.expect_request(b'bzr request 2\nmethod-name\n', b'bzr response 2\nfailed\nFooBarError\n')
    err = self.assertRaises(errors.ErrorFromSmartServer, smart_client.call, b'method-name')
    self.assertEqual((b'FooBarError',), err.error_tuple)
    medium.expect_request(b'bzr request 2\nmethod-name\n', b'bzr response 2\nsuccess\nresponse value\n')
    result = smart_client.call(b'method-name')
    self.assertEqual((b'response value',), result)
    self.assertEqual([], medium._expected_events)