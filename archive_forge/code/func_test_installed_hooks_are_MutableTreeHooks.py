from .. import mutabletree, tests
def test_installed_hooks_are_MutableTreeHooks(self):
    """The installed hooks object should be a MutableTreeHooks."""
    self.assertIsInstance(self._preserved_hooks[mutabletree.MutableTree][1], mutabletree.MutableTreeHooks)