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
def test__remote_path_relative_root(self):
    t = self.get_transport('')
    self.assertEqual('/~/', t._parsed_url.path)
    self.assertEqual('a', t._remote_path('a'))