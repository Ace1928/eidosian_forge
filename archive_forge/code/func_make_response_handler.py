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
def make_response_handler(self, response_bytes):
    from breezy.bzr.smart.message import ConventionalResponseHandler
    response_handler = ConventionalResponseHandler()
    protocol_decoder = protocol.ProtocolThreeDecoder(response_handler)
    protocol_decoder.state_accept = protocol_decoder._state_accept_expecting_message_part
    output = BytesIO()
    client_medium = medium.SmartSimplePipesClientMedium(BytesIO(response_bytes), output, 'base')
    medium_request = client_medium.get_request()
    medium_request.finished_writing()
    response_handler.setProtoAndMediumRequest(protocol_decoder, medium_request)
    return response_handler