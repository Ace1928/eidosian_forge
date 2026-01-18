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
def test_body_stream_with_empty_element_serialisation(self):
    """A body stream can include ''.

        The empty string can be transmitted like any other string.
        """
    stream = [b'', b'chunk']
    self.assertBodyStreamSerialisation(b'chunked\n' + b'0\n' + b'5\nchunk' + b'END\n', stream)
    self.assertBodyStreamRoundTrips(stream)