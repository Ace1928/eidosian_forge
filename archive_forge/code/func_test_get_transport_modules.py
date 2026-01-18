import errno
import os
import subprocess
import sys
import threading
from io import BytesIO
import breezy.transport.trace
from .. import errors, osutils, tests, transport, urlutils
from ..transport import (FileExists, NoSuchFile, UnsupportedProtocol, chroot,
from . import features, test_server
def test_get_transport_modules(self):
    handlers = transport._get_protocol_handlers()
    self.addCleanup(transport._set_protocol_handlers, handlers)
    transport._clear_protocol_handlers()

    class SampleHandler:
        """I exist, isnt that enough?"""
    transport._clear_protocol_handlers()
    transport.register_transport_proto('foo')
    transport.register_lazy_transport('foo', 'breezy.tests.test_transport', 'TestTransport.SampleHandler')
    transport.register_transport_proto('bar')
    transport.register_lazy_transport('bar', 'breezy.tests.test_transport', 'TestTransport.SampleHandler')
    self.assertEqual([SampleHandler.__module__, 'breezy.transport.chroot', 'breezy.transport.pathfilter'], transport._get_transport_modules())