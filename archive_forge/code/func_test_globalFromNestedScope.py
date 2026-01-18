import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_globalFromNestedScope(self):
    """Global names are available from nested scopes."""
    self.flakes('\n        a = 1\n        def b():\n            def c():\n                a\n        ')