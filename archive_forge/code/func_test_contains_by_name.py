import importlib.metadata as importlib_metadata
import operator
from unittest import mock
from stevedore import exception
from stevedore import extension
from stevedore.tests import utils
def test_contains_by_name(self):
    em = extension.ExtensionManager('stevedore.test.extension')
    self.assertIn('t1', em, True)