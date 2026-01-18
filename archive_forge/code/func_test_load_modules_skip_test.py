import importlib.machinery
import sys
from unittest import mock
from heat.common import plugin_loader
import heat.engine
from heat.tests import common
@mock.patch.object(plugin_loader, '_import_module', mock.MagicMock())
@mock.patch('pkgutil.walk_packages')
def test_load_modules_skip_test(self, mp):
    importer = importlib.machinery.FileFinder(heat.engine.__path__[0])
    mp.return_value = ((importer, 'hola.foo', None), (importer, 'hola.tests.test_foo', None))
    loaded = plugin_loader.load_modules(heat.engine, ignore_error=True)
    self.assertEqual(1, len(list(loaded)))