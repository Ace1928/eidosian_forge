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
def test_client_read_response_tuple_raises_UnknownSmartMethod_0_11(self):
    """read_response_tuple also raises UnknownSmartMethod if the response
        from a bzr 0.11 says the server did not recognise the request.

        (bzr 0.11 sends a slightly different error message to later versions.)
        """
    server_bytes = b"error\x01Generic bzr smart protocol error: bad request u'foo'\n"
    self._test_client_read_response_tuple_raises_UnknownSmartMethod(server_bytes)