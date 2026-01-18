import breezy
from .. import errors, lockdir, osutils, transport
from ..bzr.tests.test_smart import TestCaseWithSmartMedium
from ..lockable_files import LockableFiles, TransportLock
from ..transactions import (PassThroughTransaction, ReadOnlyTransaction,
from . import TestCaseInTempDir, TestNotApplicable
from .test_transactions import DummyWeave
def test_lock_write_returns_token_when_given_token(self):
    token = self.lockable.lock_write()
    self.addCleanup(self.lockable.unlock)
    if token is None:
        return
    new_lockable = self.get_lockable()
    token_from_new_lockable = new_lockable.lock_write(token=token)
    self.addCleanup(new_lockable.unlock)
    self.assertEqual(token, token_from_new_lockable)