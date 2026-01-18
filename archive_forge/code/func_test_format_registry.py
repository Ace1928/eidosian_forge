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
def test_format_registry(self):
    my_format_registry = self.make_format_registry()
    my_bzrdir = my_format_registry.make_controldir('lazy')
    self.assertIsInstance(my_bzrdir, DeprecatedBzrDirFormat)
    my_bzrdir = my_format_registry.make_controldir('deprecated')
    self.assertIsInstance(my_bzrdir, DeprecatedBzrDirFormat)
    my_bzrdir = my_format_registry.make_controldir('default')
    self.assertIsInstance(my_bzrdir.repository_format, knitrepo.RepositoryFormatKnit1)
    my_bzrdir = my_format_registry.make_controldir('knit')
    self.assertIsInstance(my_bzrdir.repository_format, knitrepo.RepositoryFormatKnit1)
    my_bzrdir = my_format_registry.make_controldir('branch6')
    self.assertIsInstance(my_bzrdir.get_branch_format(), breezy.bzr.branch.BzrBranchFormat6)