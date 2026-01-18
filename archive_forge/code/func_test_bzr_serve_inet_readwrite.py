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
def test_bzr_serve_inet_readwrite(self):
    self.make_branch('.')
    process, transport = self.start_server_inet(['--allow-writes'])
    branch = ControlDir.open_from_transport(transport).open_branch()
    self.make_read_requests(branch)
    transport.mkdir('adir')
    self.assertInetServerShutsdownCleanly(process)