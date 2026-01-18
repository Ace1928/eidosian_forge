import os
import socket
import sys
import time
from breezy import config, controldir, errors, tests
from breezy import transport as _mod_transport
from breezy import ui
from breezy.osutils import lexists
from breezy.tests import TestCase, TestCaseWithTransport, TestSkipped, features
from breezy.tests.http_server import HttpServer
def test_bad_connection_paramiko(self):
    """Test that a real connection attempt raises the right error"""
    from breezy.transport import ssh
    self.set_vendor(ssh.ParamikoVendor())
    t = _mod_transport.get_transport_from_url(self.bogus_url)
    self.assertRaises(errors.ConnectionError, t.get, 'foobar')