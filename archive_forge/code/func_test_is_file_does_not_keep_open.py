import sys
import unittest
import importlib_resources as resources
import pathlib
from . import data01
from . import util
from importlib import import_module
def test_is_file_does_not_keep_open(self):
    resources.files('data01').joinpath('binary.file').is_file()