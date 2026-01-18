import unittest
from fastimport import (
def test_no_paths(self):
    c = helpers.common_directory(None)
    self.assertEqual(c, None)
    c = helpers.common_directory([])
    self.assertEqual(c, None)