import sys
import unittest
import importlib_resources as resources
import pathlib
from . import data01
from . import util
from importlib import import_module
def test_is_file_exists(self):
    target = resources.files(self.data) / 'binary.file'
    self.assertTrue(target.is_file())