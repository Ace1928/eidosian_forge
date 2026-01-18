import importlib.machinery
import sys
from unittest import mock
from heat.common import plugin_loader
import heat.engine
from heat.tests import common
def test_import_module_garbage(self):
    importer = importlib.machinery.FileFinder(heat.engine.__path__[0])
    self.assertIsNone(plugin_loader._import_module(importer, 'wibble', heat.engine))