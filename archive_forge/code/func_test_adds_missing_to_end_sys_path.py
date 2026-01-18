import sys
import testtools
from fixtures import (
def test_adds_missing_to_end_sys_path(self):
    uniquedir = self.useFixture(TempDir()).path
    fixture = PythonPathEntry(uniquedir)
    self.assertFalse(uniquedir in sys.path)
    with fixture:
        self.assertTrue(uniquedir in sys.path)
    self.assertFalse(uniquedir in sys.path)