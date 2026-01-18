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
def test_name_dispatch_instance_should_build_extension_name_map(self):
    extensions = [test_extension, test_extension2]
    em = NameDispatchExtensionManager.make_test_instance(extensions)
    self.assertEqual(test_extension, em.by_name[test_extension.name])
    self.assertEqual(test_extension2, em.by_name[test_extension2.name])