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
def make_response(self, args, body=None, body_stream=None):
    response_io = BytesIO()
    response = _mod_request.SuccessfulSmartServerResponse(args, body=body, body_stream=body_stream)
    responder = protocol.ProtocolThreeResponder(response_io.write)
    responder.send_response(response)
    return response_io.getvalue()