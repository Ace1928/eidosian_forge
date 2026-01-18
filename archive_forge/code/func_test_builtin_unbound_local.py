from pyflakes import messages as m
from pyflakes.test.harness import TestCase
def test_builtin_unbound_local(self):
    self.flakes('\n        def foo():\n            a = range(1, 10)\n            range = a\n            return range\n\n        foo()\n\n        print(range)\n        ', m.UndefinedLocal)