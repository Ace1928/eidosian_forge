from io import BytesIO
from dulwich.tests import TestCase
from ..objects import ZERO_SHA
from ..reflog import (
def test_drop_entry(self):
    drop_reflog_entry(self.f, 0)
    log = self._read_log()
    self.assertEqual(len(log), 2)
    self.assertEqual(self.original_log[0:2], log)
    self.f.seek(0)
    drop_reflog_entry(self.f, 1)
    log = self._read_log()
    self.assertEqual(len(log), 1)
    self.assertEqual(self.original_log[1], log[0])