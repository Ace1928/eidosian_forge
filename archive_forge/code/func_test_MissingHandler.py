from unittest import TestCase
from fastimport import (
def test_MissingHandler(self):
    e = errors.MissingHandler('foo')
    self.assertEqual('Missing handler for command foo', str(e))