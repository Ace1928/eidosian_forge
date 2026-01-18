import breezy
from .. import errors, lockdir, osutils, transport
from ..bzr.tests.test_smart import TestCaseWithSmartMedium
from ..lockable_files import LockableFiles, TransportLock
from ..transactions import (PassThroughTransaction, ReadOnlyTransaction,
from . import TestCaseInTempDir, TestNotApplicable
from .test_transactions import DummyWeave
def test__file_modes(self):
    self.transport.mkdir('readonly')
    osutils.make_readonly('readonly')
    lockable = LockableFiles(self.transport.clone('readonly'), 'test-lock', lockdir.LockDir)
    self.assertEqual(448, lockable._dir_mode & 448)
    self.assertEqual(384, lockable._file_mode & 448)