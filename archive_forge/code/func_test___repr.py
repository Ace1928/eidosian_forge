from .. import branch, errors
from .. import hooks as _mod_hooks
from .. import pyutils, tests
from ..hooks import (HookPoint, Hooks, UnknownHook, install_lazy_named_hook,
def test___repr(self):
    hook = HookPoint('foo', 'no docs', None, None)

    def callback():
        pass
    hook.hook(callback, 'my callback')
    callback_repr = repr(callback)
    self.assertEqual('<HookPoint(foo), callbacks=[%s(my callback)]>' % callback_repr, repr(hook))