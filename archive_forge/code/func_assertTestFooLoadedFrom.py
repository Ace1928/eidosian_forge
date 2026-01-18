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
def assertTestFooLoadedFrom(self, path):
    self.assertPluginKnown('test_foo')
    self.assertDocstring('This is the doc for test_foo', self.module.test_foo)
    self.assertEqual(path, self.module.test_foo.dir_source)