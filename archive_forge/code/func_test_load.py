import tempfile
import unittest
from pathlib import Path
from bpython.config import getpreferredencoding
from bpython.history import History
def test_load(self):
    history = History()
    history.load(self.filename, self.encoding)
    self.assertEqual(history.entries, ['#1', '#2'])