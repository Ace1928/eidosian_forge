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
def test_connection_vendor(self):
    raise TestSkipped("We don't test spawning real ssh, because it prompts for a password. Enable this test if we figure out how to prevent this.")
    self.set_vendor(None)
    t = self.get_transport()
    self.assertEqual(b'foobar\n', t.get('a_file').read())