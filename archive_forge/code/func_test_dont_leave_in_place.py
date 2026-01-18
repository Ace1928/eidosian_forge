import breezy
from .. import errors, lockdir, osutils, transport
from ..bzr.tests.test_smart import TestCaseWithSmartMedium
from ..lockable_files import LockableFiles, TransportLock
from ..transactions import (PassThroughTransaction, ReadOnlyTransaction,
from . import TestCaseInTempDir, TestNotApplicable
from .test_transactions import DummyWeave
def test_dont_leave_in_place(self):
    token = self.lockable.lock_write()
    try:
        if token is None:
            return
        self.lockable.leave_in_place()
    finally:
        self.lockable.unlock()
    new_lockable = self.get_lockable()
    new_lockable.lock_write(token=token)
    new_lockable.dont_leave_in_place()
    new_lockable.unlock()
    third_lockable = self.get_lockable()
    third_lockable.lock_write()
    third_lockable.unlock()