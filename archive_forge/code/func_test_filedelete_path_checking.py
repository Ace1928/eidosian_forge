from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
def test_filedelete_path_checking(self):
    self.assertRaises(ValueError, commands.FileDeleteCommand, b'')
    self.assertRaises(ValueError, commands.FileDeleteCommand, None)