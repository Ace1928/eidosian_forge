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
def test_named_dispatch_map_should_invoke_filter_for_extensions(self):
    em = NameDispatchExtensionManager.make_test_instance([test_extension, test_extension2])
    func = Mock()
    args = ('A',)
    kw = {'BIGGER': 'Cheese'}
    em.map(['test_extension'], func, *args, **kw)
    func.assert_called_once_with(test_extension, *args, **kw)