import os
import unittest
from distutils.text_file import TextFile
from distutils.tests import support
def test_input(count, description, file, expected_result):
    result = file.readlines()
    self.assertEqual(result, expected_result)