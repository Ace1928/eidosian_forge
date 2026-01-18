import importlib.metadata as importlib_metadata
import operator
from unittest import mock
from stevedore import exception
from stevedore import extension
from stevedore.tests import utils
def test_load_multiple_times_plugins(self):
    em1 = extension.ExtensionManager('stevedore.test.extension')
    plugins1 = [ext.plugin for ext in em1]
    em2 = extension.ExtensionManager('stevedore.test.extension')
    plugins2 = [ext.plugin for ext in em2]
    self.assertIs(plugins1[0], plugins2[0])