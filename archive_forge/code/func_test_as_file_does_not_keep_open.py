import sys
import unittest
import importlib_resources as resources
import pathlib
from . import data01
from . import util
from importlib import import_module
@unittest.skip('Desired but not supported.')
def test_as_file_does_not_keep_open(self):
    resources.as_file(resources.files('data01') / 'binary.file')