from unittest import TestCase
from fastimport import (
def test_UnknownDateFormat(self):
    e = errors.UnknownDateFormat('aaa')
    self.assertEqual("Unknown date format 'aaa'", str(e))