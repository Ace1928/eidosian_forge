import importlib.metadata as importlib_metadata
import operator
from unittest import mock
from stevedore import exception
from stevedore import extension
from stevedore.tests import utils
def test_list_entry_points_names(self):
    em = extension.ExtensionManager('stevedore.test.extension')
    names = em.entry_points_names()
    self.assertEqual(set(['e1', 'e2', 't1', 't2']), set(names))
    self.assertEqual(4, len(names))