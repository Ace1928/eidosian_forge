import tempfile
import unittest
from pathlib import Path
from bpython.config import getpreferredencoding
from bpython.history import History
def test_forward(self):
    self.history.first()
    self.assertEqual(self.history.forward(), '#1')
    self.assertNotEqual(self.history.forward(), '#1')
    self.assertEqual(self.history.forward(), '#3')
    for x in range(1000 - 4 - 1):
        self.history.forward()
    self.assertEqual(self.history.forward(), '#999')