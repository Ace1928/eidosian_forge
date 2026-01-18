import io
import unittest
import importlib_resources as resources
from importlib_resources._adapters import (
from . import util
def test_child_path_iter(self):
    self.assertEqual(list((self.files / 'a').iterdir()), [])