from unittest.mock import Mock
from unittest.mock import sentinel
from stevedore.dispatch import DispatchExtensionManager
from stevedore.dispatch import NameDispatchExtensionManager
from stevedore.extension import Extension
from stevedore.tests import utils
from stevedore import DriverManager
from stevedore import EnabledExtensionManager
from stevedore import ExtensionManager
from stevedore import HookManager
from stevedore import NamedExtensionManager
def test_instance_call(self):

    def invoke(ext, *args, **kwds):
        return (ext.name, args, kwds)
    em = DriverManager.make_test_instance(a_driver)
    result = em(invoke, 'a', b='C')
    self.assertEqual(result, ('test_driver', ('a',), {'b': 'C'}))