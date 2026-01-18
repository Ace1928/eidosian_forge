import breezy
from .. import errors, lockdir, osutils, transport
from ..bzr.tests.test_smart import TestCaseWithSmartMedium
from ..lockable_files import LockableFiles, TransportLock
from ..transactions import (PassThroughTransaction, ReadOnlyTransaction,
from . import TestCaseInTempDir, TestNotApplicable
from .test_transactions import DummyWeave
def get_lockable(self):
    branch = breezy.branch.Branch.open(self.get_url('foo'))
    self.addCleanup(branch.controldir.transport.disconnect)
    return branch.control_files