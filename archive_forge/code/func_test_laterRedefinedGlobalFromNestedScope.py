import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_laterRedefinedGlobalFromNestedScope(self):
    """
        Test that referencing a local name that shadows a global, before it is
        defined, generates a warning.
        """
    self.flakes('\n        a = 1\n        def fun():\n            a\n            a = 2\n            return a\n        ', m.UndefinedLocal)