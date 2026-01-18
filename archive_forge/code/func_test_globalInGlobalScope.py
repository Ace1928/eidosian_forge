import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_globalInGlobalScope(self):
    """
        A global statement in the global scope is ignored.
        """
    self.flakes('\n        global x\n        def foo():\n            print(x)\n        ', m.UndefinedName)