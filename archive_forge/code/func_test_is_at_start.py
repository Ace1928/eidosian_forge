import tempfile
import unittest
from pathlib import Path
from bpython.config import getpreferredencoding
from bpython.history import History
def test_is_at_start(self):
    self.history.first()
    self.assertNotEqual(self.history.index, 0)
    self.assertTrue(self.history.is_at_end)
    self.history.forward()
    self.assertFalse(self.history.is_at_end)