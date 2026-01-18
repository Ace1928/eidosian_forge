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
def test_connection_paramiko(self):
    from breezy.transport import ssh
    self.set_vendor(ssh.ParamikoVendor())
    t = self.get_transport()
    self.assertEqual(b'foobar\n', t.get('a_file').read())