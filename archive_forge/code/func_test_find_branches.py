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
def test_find_branches(self):
    root = self.make_repository('', shared=True)
    foo, bar, baz = self.make_foo_bar_baz()
    qux = self.make_controldir('foo/qux')
    t = self.get_transport()
    branches = bzrdir.BzrDir.find_branches(t)
    self.assertEqual(baz.root_transport.base, branches[0].base)
    self.assertEqual(foo.root_transport.base, branches[1].base)
    self.assertEqual(bar.root_transport.base, branches[2].base)
    branches = bzrdir.BzrDir.find_branches(t.clone('foo'))
    self.assertEqual(foo.root_transport.base, branches[0].base)
    self.assertEqual(bar.root_transport.base, branches[1].base)