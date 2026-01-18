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
def test_multiple_bytes_parts(self):
    """Each bytes part triggers a call to the request_handler's
        accept_body method.
        """
    multiple_bytes_parts = b's\x00\x00\x00\x07l3:fooeb\x00\x00\x00\x0bSome bytes\nb\x00\x00\x00\nMore bytese'
    request_handler = self.make_request_handler(multiple_bytes_parts)
    accept_body_calls = [call_info[1] for call_info in request_handler.calls if call_info[0] == 'accept_body']
    self.assertEqual([b'Some bytes\n', b'More bytes'], accept_body_calls)