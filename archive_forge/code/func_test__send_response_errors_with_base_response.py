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
def test__send_response_errors_with_base_response(self):
    """Ensure that only the Successful/Failed subclasses are used."""
    smart_protocol = self.server_protocol_class(None, lambda x: None)
    self.assertRaises(AttributeError, smart_protocol._send_response, _mod_request.SmartServerResponse((b'x',)))