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
def test_determine_stacking_policy_relative(self):
    parent_bzrdir = self.make_controldir('.')
    child_bzrdir = self.make_controldir('child')
    parent_bzrdir.get_config().set_default_stack_on('child2')
    repo_policy = child_bzrdir.determine_repository_policy()
    self.assertEqual('child2', repo_policy._stack_on)
    self.assertEqual(parent_bzrdir.root_transport.base, repo_policy._stack_on_pwd)