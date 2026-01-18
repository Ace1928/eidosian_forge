import importlib
import logging
import os
import sys
import types
from io import StringIO
from typing import Any, Dict, List
import breezy
from .. import osutils, plugin, tests
from . import test_bar
def create_plugin_package(self, name, dir=None, source=None):
    if dir is None:
        dir = name
    if source is None:
        source = '"""This is the doc for {}"""\ndir_source = \'{}\'\n'.format(name, dir)
    os.makedirs(dir)
    self.create_plugin(name, source, dir, file_name='__init__.py')