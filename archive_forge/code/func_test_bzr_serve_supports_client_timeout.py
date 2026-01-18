import signal
import sys
import threading
from _thread import interrupt_main  # type: ignore
from ... import builtins, config, errors, osutils
from ... import revision as _mod_revision
from ... import trace, transport, urlutils
from ...branch import Branch
from ...bzr.smart import client, medium
from ...bzr.smart.server import BzrServerFactory, SmartTCPServer
from ...controldir import ControlDir
from ...transport import remote
from .. import TestCaseWithMemoryTransport, TestCaseWithTransport
def test_bzr_serve_supports_client_timeout(self):
    process, url = self.start_server_port(['--client-timeout=0.1'])
    self.build_tree_contents([('a_file', b'contents\n')])
    t = transport.get_transport_from_url(url)
    self.assertEqual(b'contents\n', t.get_bytes('a_file'))
    m = t.get_smart_medium()
    m.read_bytes(1)
    err = process.stderr.readline()
    self.assertEqual(b'Connection Timeout: disconnecting client after 0.1 seconds\n', err)
    self.assertServerFinishesCleanly(process)