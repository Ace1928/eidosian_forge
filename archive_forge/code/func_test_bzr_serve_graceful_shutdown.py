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
def test_bzr_serve_graceful_shutdown(self):
    big_contents = b'a' * 64 * 1024
    self.build_tree_contents([('bigfile', big_contents)])
    process, url = self.start_server_port(['--client-timeout=1.0'])
    t = transport.get_transport_from_url(url)
    m = t.get_smart_medium()
    c = client._SmartClient(m)
    resp, response_handler = c.call_expecting_body(b'get', b'bigfile')
    self.assertEqual((b'ok',), resp)
    process.send_signal(signal.SIGHUP)
    self.assertEqual(b'Requested to stop gracefully\n', process.stderr.readline())
    self.assertIn(process.stderr.readline(), (b'', b'Waiting for 1 client(s) to finish\n'))
    body = response_handler.read_body_bytes()
    if body != big_contents:
        self.fail('Failed to properly read the contents of "bigfile"')
    self.assertEqual(b'', m.read_bytes(1))
    self.assertEqual(0, process.wait())