from unittest import TestCase
from fastimport import (
def test_UnknownFeature(self):
    e = errors.UnknownFeature('aaa')
    self.assertEqual("Unknown feature 'aaa' - try a later importer or an earlier data format", str(e))