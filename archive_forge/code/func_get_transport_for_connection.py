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
def get_transport_for_connection(self, set_config):
    port = self.get_server().port
    if set_config:
        conf = config.AuthenticationConfig()
        conf._get_config().update({'sftptest': {'scheme': 'ssh', 'port': port, 'user': 'bar'}})
        conf._save()
    t = _mod_transport.get_transport_from_url('sftp://localhost:%d' % port)
    t.has('foo')
    return t