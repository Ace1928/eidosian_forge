import tempfile
import unittest
from pathlib import Path
from bpython.config import getpreferredencoding
from bpython.history import History
def test_enter(self):
    self.history.enter('#lastnumber!')
    self.assertEqual(self.history.back(), '#lastnumber!')
    self.assertEqual(self.history.forward(), '#lastnumber!')