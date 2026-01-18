from .. import branch, errors
from .. import hooks as _mod_hooks
from .. import pyutils, tests
from ..hooks import (HookPoint, Hooks, UnknownHook, install_lazy_named_hook,
def test_install_named_hook_appends_known_hook(self):
    hooks = Hooks('breezy.tests.test_hooks', 'some_hooks')
    hooks['set_rh'] = []
    hooks.install_named_hook('set_rh', None, 'demo')
    self.assertEqual(hooks['set_rh'], [None])