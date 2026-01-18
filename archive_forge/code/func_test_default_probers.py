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
def test_default_probers(self):
    self.addCleanup(ControlDirFormat.unregister_prober, TrackingProber)
    ControlDirFormat.register_prober(TrackingProber)
    TrackingProber.seen_urls = []
    opener = self.make_branch_opener(['.'], probers=[TrackingProber])
    self.assertRaises(NotBranchError, opener.open, '.')
    self.assertEqual(1, len(TrackingProber.seen_urls))
    TrackingProber.seen_urls = []
    self.assertRaises(NotBranchError, ControlDir.open, '.')
    self.assertEqual(1, len(TrackingProber.seen_urls))