from .. import branch, errors
from .. import hooks as _mod_hooks
from .. import pyutils, tests
from ..hooks import (HookPoint, Hooks, UnknownHook, install_lazy_named_hook,
def test_install_lazy_named_hook(self):
    self.hooks.add_hook('set_rh', 'doc', (0, 15))

    def set_rh():
        return None
    install_lazy_named_hook('breezy.tests.test_hooks', 'TestHooks.hooks', 'set_rh', set_rh, 'demo')
    set_rh_lazy_hooks = _mod_hooks._lazy_hooks['breezy.tests.test_hooks', 'TestHooks.hooks', 'set_rh']
    self.assertEqual(1, len(set_rh_lazy_hooks))
    self.assertEqual(set_rh, set_rh_lazy_hooks[0][0].get_obj())
    self.assertEqual('demo', set_rh_lazy_hooks[0][1])
    self.assertEqual(list(TestHooks.hooks['set_rh']), [set_rh])