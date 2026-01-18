import importlib.metadata as importlib_metadata
import operator
from unittest import mock
from stevedore import exception
from stevedore import extension
from stevedore.tests import utils
def test_map_return_values(self):

    def mapped(ext, *args, **kwds):
        return ext.name
    em = extension.ExtensionManager('stevedore.test.extension', invoke_on_load=True)
    results = em.map(mapped)
    self.assertEqual(sorted(results), WORKING_NAMES)