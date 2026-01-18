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
def test_error_flag_after_body(self):
    body_then_error = b's\x00\x00\x00\x07l3:fooeb\x00\x00\x00\x0bSome bytes\nb\x00\x00\x00\nMore bytesoEs\x00\x00\x00\x07l3:baree'
    request_handler = self.make_request_handler(body_then_error)
    self.assertEqual([('post_body_error_received', (b'bar',)), ('end_received',)], request_handler.calls[-2:])