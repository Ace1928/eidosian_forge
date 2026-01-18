from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_nestedFunctionsNestScope(self):
    self.flakes('\n        def a():\n            def b():\n                fu\n            import fu\n        ')