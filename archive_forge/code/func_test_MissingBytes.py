from unittest import TestCase
from fastimport import (
def test_MissingBytes(self):
    e = errors.MissingBytes(99, 10, 8)
    self.assertEqual('line 99: Unexpected EOF - expected 10 bytes, found 8', str(e))