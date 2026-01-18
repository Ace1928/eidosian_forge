from .. import branch, errors
from .. import hooks as _mod_hooks
from .. import pyutils, tests
from ..hooks import (HookPoint, Hooks, UnknownHook, install_lazy_named_hook,
def test_install_named_hook_and_retrieve_name(self):
    hooks = Hooks('breezy.tests.test_hooks', 'somehooks')
    hooks['set_rh'] = []
    hooks.install_named_hook('set_rh', None, 'demo')
    self.assertEqual('demo', hooks.get_hook_name(None))