from unittest import TestCase
from fastimport import (
def test_BadFormat(self):
    e = errors.BadFormat(99, 'foo', 'bar', 'xyz')
    self.assertEqual("line 99: Bad format for section bar in command foo: found 'xyz'", str(e))