import tempfile
import unittest
from pathlib import Path
from bpython.config import getpreferredencoding
from bpython.history import History
def test_first(self):
    self.history.first()
    self.assertFalse(self.history.is_at_start)
    self.assertTrue(self.history.is_at_end)