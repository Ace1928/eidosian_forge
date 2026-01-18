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
def test_allowed_reference(self):
    opener = self.make_branch_opener(True, ['a', 'b', None])
    self.assertEqual('b', opener.check_and_follow_branch_reference('a'))
    self.assertEqual(['a', 'b'], opener.follow_reference_calls)