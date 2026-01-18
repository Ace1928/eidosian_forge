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
def test_acquire_repository_standalone(self):
    """The default acquisition policy should create a standalone branch."""
    my_bzrdir = self.make_controldir('.')
    repo_policy = my_bzrdir.determine_repository_policy()
    repo, is_new = repo_policy.acquire_repository()
    self.assertEqual(repo.controldir.root_transport.base, my_bzrdir.root_transport.base)
    self.assertFalse(repo.is_shared())