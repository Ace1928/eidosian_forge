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
def test__send_no_retry_pipes(self):
    client_read, server_write = create_file_pipes()
    server_read, client_write = create_file_pipes()
    client_medium = medium.SmartSimplePipesClientMedium(client_read, client_write, base='/')
    smart_client = client._SmartClient(client_medium)
    smart_request = client._SmartClientRequest(smart_client, b'hello', ())
    server_read.close()
    encoder, response_handler = smart_request._construct_protocol(3)
    self.assertRaises(errors.ConnectionReset, smart_request._send_no_retry, encoder)