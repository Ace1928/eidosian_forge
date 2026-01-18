import importlib.machinery
import sys
from unittest import mock
from heat.common import plugin_loader
import heat.engine
from heat.tests import common
def test_create_subpackage_single_path(self):
    pkg_name = 'heat.engine.test_single_path'
    self.assertNotIn(pkg_name, sys.modules)
    pkg = plugin_loader.create_subpackage('/tmp', 'heat.engine', 'test_single_path')
    self.assertIn(pkg_name, sys.modules)
    self.assertEqual(sys.modules[pkg_name], pkg)
    self.assertEqual(['/tmp'], pkg.__path__)
    self.assertEqual(pkg_name, pkg.__name__)