from pyflakes import messages as m
from pyflakes.test.harness import TestCase
def test_global_shadowing_builtin(self):
    self.flakes('\n        def f():\n            global range\n            range = None\n            print(range)\n\n        f()\n        ')