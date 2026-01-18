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
def make_read_requests(self, branch):
    """Do some read only requests."""
    with branch.lock_read():
        branch.repository.all_revision_ids()
        self.assertEqual(_mod_revision.NULL_REVISION, branch.last_revision())