import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_delGlobal(self):
    """Del a global binding from a function."""
    self.flakes('\n        a = 1\n        def f():\n            global a\n            del a\n        a\n        ')