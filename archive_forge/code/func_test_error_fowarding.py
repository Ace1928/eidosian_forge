import pathlib
import unittest
from pygame import encode_string, encode_file_path
def test_error_fowarding(self):
    self.assertRaises(SyntaxError, encode_file_path)