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
def make_client_protocol_and_output(self, input_bytes=None):
    """
        :returns: a Request
        """
    if input_bytes is None:
        input = BytesIO()
    else:
        input = BytesIO(input_bytes)
    output = BytesIO()
    client_medium = medium.SmartSimplePipesClientMedium(input, output, 'base')
    request = client_medium.get_request()
    if self.client_protocol_class is not None:
        client_protocol = self.client_protocol_class(request)
        return (client_protocol, client_protocol, output)
    else:
        self.assertNotEqual(None, self.request_encoder)
        self.assertNotEqual(None, self.response_decoder)
        requester = self.request_encoder(request)
        response_handler = message.ConventionalResponseHandler()
        response_protocol = self.response_decoder(response_handler, expect_version_marker=True)
        response_handler.setProtoAndMediumRequest(response_protocol, request)
        return (requester, response_handler, output)