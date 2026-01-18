import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_laterRedefinedGlobalFromNestedScope3(self):
    """
        Test that referencing a local name in a nested scope that shadows a
        global, before it is defined, generates a warning.
        """
    self.flakes('\n            def fun():\n                a = 1\n                def fun2():\n                    a\n                    a = 1\n                    return a\n                return a\n        ', m.UndefinedLocal)