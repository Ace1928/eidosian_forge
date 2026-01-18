from .. import branch, errors
from .. import hooks as _mod_hooks
from .. import pyutils, tests
from ..hooks import (HookPoint, Hooks, UnknownHook, install_lazy_named_hook,
def test_known_hooks_key_to_parent_and_attribute(self):
    self.assertEqual((branch.Branch, 'hooks'), known_hooks.key_to_parent_and_attribute(('breezy.branch', 'Branch.hooks')))
    self.assertEqual((branch, 'Branch'), known_hooks.key_to_parent_and_attribute(('breezy.branch', 'Branch')))