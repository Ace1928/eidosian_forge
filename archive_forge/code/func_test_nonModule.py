import sys
from types import ModuleType
from twisted.trial.unittest import TestCase
def test_nonModule(self):
    """
        A non-C{dict} value in the attributes dictionary passed to L{_makePackages}
        is preserved unchanged in the return value.
        """
    modules = {}
    _makePackages(None, dict(reactor='reactor'), modules)
    self.assertEqual(modules, dict(reactor='reactor'))