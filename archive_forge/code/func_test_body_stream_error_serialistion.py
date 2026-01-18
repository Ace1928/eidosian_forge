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
def test_body_stream_error_serialistion(self):
    stream = [b'first chunk', _mod_request.FailedSmartServerResponse((b'FailureName', b'failure arg'))]
    expected_bytes = b'chunked\n' + b'b\nfirst chunk' + b'ERR\n' + b'b\nFailureName' + b'b\nfailure arg' + b'END\n'
    self.assertBodyStreamSerialisation(expected_bytes, stream)
    self.assertBodyStreamRoundTrips(stream)