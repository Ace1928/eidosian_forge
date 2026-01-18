import importlib.metadata as importlib_metadata
import operator
from unittest import mock
from stevedore import exception
from stevedore import extension
from stevedore.tests import utils
def test_entry_point_target(self):
    self.assertEqual('module.name:attribute.name [extra]', self.ext1.entry_point_target)
    self.assertEqual('module:attribute', self.ext2.entry_point_target)