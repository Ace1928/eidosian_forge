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
def test_bzr_serve_quiet(self):
    self.make_branch('.')
    args = ['--listen', 'localhost', '--port', '0', '--quiet']
    out, err = self.run_bzr_serve_then_func(args, retcode=3)
    self.assertEqual('', out)
    self.assertEqual('', err)