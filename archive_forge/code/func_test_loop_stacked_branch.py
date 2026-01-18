from typing import List
from .. import urlutils
from ..branch import Branch
from ..bzr import BzrProber
from ..bzr.branch import BranchReferenceFormat
from ..controldir import ControlDir, ControlDirFormat
from ..errors import NotBranchError, RedirectRequested
from ..transport import (Transport, chroot, get_transport, register_transport,
from ..url_policy_open import (BadUrl, BranchLoopError, BranchOpener,
from . import TestCase, TestCaseWithTransport
def test_loop_stacked_branch(self):
    a = self.make_branch('a', format='1.6')
    b = self.make_branch('b', format='1.6')
    a.set_stacked_on_url(b.base)
    b.set_stacked_on_url(a.base)
    opener = self.make_branch_opener([a.base, b.base])
    self.assertRaises(BranchLoopError, opener.open, a.base)
    self.assertRaises(BranchLoopError, opener.open, b.base)