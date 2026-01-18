import breezy
from .. import errors, lockdir, osutils, transport
from ..bzr.tests.test_smart import TestCaseWithSmartMedium
from ..lockable_files import LockableFiles, TransportLock
from ..transactions import (PassThroughTransaction, ReadOnlyTransaction,
from . import TestCaseInTempDir, TestNotApplicable
from .test_transactions import DummyWeave
def test_lock_write_raises_on_token_mismatch(self):
    token = self.lockable.lock_write()
    self.addCleanup(self.lockable.unlock)
    if token is None:
        return
    different_token = token + b'xxx'
    self.assertRaises(errors.TokenMismatch, self.lockable.lock_write, token=different_token)
    new_lockable = self.get_lockable()
    self.assertRaises(errors.TokenMismatch, new_lockable.lock_write, token=different_token)