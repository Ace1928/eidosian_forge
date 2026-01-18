from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_nestedClassAndFunctionScope(self):
    self.flakes('\n        def a():\n            import fu\n            class b:\n                def c(self):\n                    print(fu)\n        ')