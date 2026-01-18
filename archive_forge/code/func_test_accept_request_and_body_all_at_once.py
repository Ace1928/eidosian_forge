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
def test_accept_request_and_body_all_at_once(self):
    self.overrideEnv('BRZ_NO_SMART_VFS', None)
    mem_transport = memory.MemoryTransport()
    mem_transport.put_bytes('foo', b'abcdefghij')
    out_stream = BytesIO()
    smart_protocol = self.server_protocol_class(mem_transport, out_stream.write)
    smart_protocol.accept_bytes(b'readv\x01foo\n3\n3,3done\n')
    self.assertEqual(0, smart_protocol.next_read_size())
    self.assertEqual(self.response_marker + b'success\nreadv\n3\ndefdone\n', out_stream.getvalue())
    self.assertEqual(b'', smart_protocol.unused_data)
    self.assertEqual(b'', smart_protocol.in_buffer)