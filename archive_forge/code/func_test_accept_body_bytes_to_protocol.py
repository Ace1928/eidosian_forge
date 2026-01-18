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
def test_accept_body_bytes_to_protocol(self):
    protocol = self.build_protocol_waiting_for_body()
    self.assertEqual(6, protocol.next_read_size())
    protocol.accept_bytes(b'7\nabc')
    self.assertEqual(9, protocol.next_read_size())
    protocol.accept_bytes(b'defgd')
    protocol.accept_bytes(b'one\n')
    self.assertEqual(0, protocol.next_read_size())
    self.assertTrue(self.end_received)