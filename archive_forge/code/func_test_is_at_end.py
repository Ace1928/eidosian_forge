import tempfile
import unittest
from pathlib import Path
from bpython.config import getpreferredencoding
from bpython.history import History
def test_is_at_end(self):
    self.history.last()
    self.assertEqual(self.history.index, 0)
    self.assertTrue(self.history.is_at_start)
    self.assertFalse(self.history.is_at_end)