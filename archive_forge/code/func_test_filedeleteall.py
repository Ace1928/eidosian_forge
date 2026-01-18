from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
def test_filedeleteall(self):
    c = commands.FileDeleteAllCommand()
    self.assertEqual(b'deleteall', bytes(c))