from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
def test_blob(self):
    c = commands.BlobCommand(b'1', b'hello world')
    self.assertEqual(b'blob\nmark :1\ndata 11\nhello world', bytes(c))