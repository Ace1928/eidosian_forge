import os
import unittest
from unittest import mock
def test_add_module(self):
    self.module._add_module('something/test.py')
    self.assertIn(os.path.abspath('something/test'), self.module.dirs[os.path.abspath('something')])