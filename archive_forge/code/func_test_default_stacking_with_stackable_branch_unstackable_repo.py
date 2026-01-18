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
def test_default_stacking_with_stackable_branch_unstackable_repo(self):
    source_bzrdir = self.make_controldir('source')
    knitpack_repo.RepositoryFormatKnitPack1().initialize(source_bzrdir)
    source_branch = breezy.bzr.branch.BzrBranchFormat7().initialize(source_bzrdir)
    parent_bzrdir = self.make_controldir('parent')
    stacked_on = self.make_branch('parent/stacked-on', format='pack-0.92')
    parent_bzrdir.get_config().set_default_stack_on(stacked_on.base)
    target = source_bzrdir.clone(self.get_url('parent/target'))