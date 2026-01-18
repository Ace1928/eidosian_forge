import sys
import unittest
import importlib_resources as resources
import pathlib
from . import data01
from . import util
from importlib import import_module
def test_iterdir_does_not_keep_open(self):
    [item.name for item in resources.files('data01').iterdir()]