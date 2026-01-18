import tempfile
import unittest
from pathlib import Path
from bpython.config import getpreferredencoding
from bpython.history import History
def test_append_reload_and_write(self):
    history = History()
    history.append_reload_and_write('#3', self.filename, self.encoding)
    self.assertEqual(history.entries, ['#1', '#2', '#3'])
    history.append_reload_and_write('#4', self.filename, self.encoding)
    self.assertEqual(history.entries, ['#1', '#2', '#3', '#4'])