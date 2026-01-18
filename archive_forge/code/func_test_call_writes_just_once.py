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
def test_call_writes_just_once(self):
    """A bodyless request is written to the medium all at once."""
    medium_request = StubMediumRequest()
    encoder = protocol.ProtocolThreeRequester(medium_request)
    encoder.call(b'arg1', b'arg2', b'arg3')
    self.assertEqual(['accept_bytes', 'finished_writing'], medium_request.calls)