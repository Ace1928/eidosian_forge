import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_break_lock_corrupt_info(self):
    """break_lock works even if the info file is corrupt (and tells the UI
        that it is corrupt).
        """
    ld = self.get_lock()
    ld2 = self.get_lock()
    ld.create()
    ld.lock_write()
    ld.transport.put_bytes_non_atomic('test_lock/held/info', b'\x00')

    class LoggingUIFactory(breezy.ui.SilentUIFactory):

        def __init__(self):
            self.prompts = []

        def get_boolean(self, prompt):
            self.prompts.append(('boolean', prompt))
            return True
    ui = LoggingUIFactory()
    self.overrideAttr(breezy.ui, 'ui_factory', ui)
    ld2.break_lock()
    self.assertLength(1, ui.prompts)
    self.assertEqual('boolean', ui.prompts[0][0])
    self.assertStartsWith(ui.prompts[0][1], 'Break (corrupt LockDir')
    self.assertRaises(LockBroken, ld.unlock)