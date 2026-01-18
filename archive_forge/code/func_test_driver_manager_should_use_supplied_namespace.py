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
def test_driver_manager_should_use_supplied_namespace(self):
    namespace = 'testing.1.2.3'
    em = DriverManager.make_test_instance(a_driver, namespace=namespace)
    self.assertEqual(namespace, em.namespace)