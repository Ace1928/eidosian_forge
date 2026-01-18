import io
import unittest
import importlib_resources as resources
from importlib_resources._adapters import (
from . import util
def test_child_path_is(self):
    self.assertTrue((self.files / 'a').is_file())
    self.assertFalse((self.files / 'a').is_dir())