from .. import branch, errors
from .. import hooks as _mod_hooks
from .. import pyutils, tests
from ..hooks import (HookPoint, Hooks, UnknownHook, install_lazy_named_hook,
def test_hook(self):
    hook = HookPoint('foo', 'no docs', None, None)

    def callback():
        pass
    hook.hook(callback, 'my callback')
    self.assertEqual([callback], list(hook))