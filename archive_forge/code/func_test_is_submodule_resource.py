import sys
import unittest
import importlib_resources as resources
import pathlib
from . import data01
from . import util
from importlib import import_module
def test_is_submodule_resource(self):
    self.assertTrue(resources.files(import_module('namespacedata01')).joinpath('binary.file').is_file())