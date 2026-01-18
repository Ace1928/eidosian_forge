import breezy
from .. import errors, lockdir, osutils, transport
from ..bzr.tests.test_smart import TestCaseWithSmartMedium
from ..lockable_files import LockableFiles, TransportLock
from ..transactions import (PassThroughTransaction, ReadOnlyTransaction,
from . import TestCaseInTempDir, TestNotApplicable
from .test_transactions import DummyWeave
def test_lock_created(self):
    self.assertTrue(self.transport.has('my-lock'))
    self.lockable.lock_write()
    self.assertTrue(self.transport.has('my-lock/held/info'))
    self.lockable.unlock()
    self.assertFalse(self.transport.has('my-lock/held/info'))
    self.assertTrue(self.transport.has('my-lock'))