import os
import subprocess
import sys
import breezy.branch
import breezy.bzr.branch
from ... import (branch, bzr, config, controldir, errors, help_topics, lock,
from ... import revision as _mod_revision
from ... import transport as _mod_transport
from ... import urlutils, win32utils
from ...errors import (NotBranchError, UnknownFormatError,
from ...tests import (TestCase, TestCaseWithMemoryTransport,
from ...transport import memory, pathfilter
from ...transport.http.urllib import HttpTransport
from ...transport.nosmart import NoSmartTransportDecorator
from ...transport.readonly import ReadonlyTransportDecorator
from .. import branch as bzrbranch
from .. import (bzrdir, knitpack_repo, knitrepo, remote, workingtree_3,
from ..fullhistory import BzrBranchFormat5
def test_initialize_on_format_uses_smart_transport(self):
    self.setup_smart_server_with_call_log()
    new_format = controldir.format_registry.make_controldir('dirstate')
    transport = self.get_transport('target')
    transport.ensure_base()
    self.reset_smart_call_log()
    instance = new_format.initialize_on_transport(transport)
    self.assertIsInstance(instance, remote.RemoteBzrDir)
    rpc_count = len(self.hpss_calls)
    self.assertEqual(2, rpc_count)