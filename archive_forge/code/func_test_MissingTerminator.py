from unittest import TestCase
from fastimport import (
def test_MissingTerminator(self):
    e = errors.MissingTerminator(99, '---')
    self.assertEqual("line 99: Unexpected EOF - expected '---' terminator", str(e))