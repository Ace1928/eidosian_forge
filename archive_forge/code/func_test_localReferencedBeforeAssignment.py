from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_localReferencedBeforeAssignment(self):
    self.flakes('\n        a = 1\n        def f():\n            a; a=1\n        f()\n        ', m.UndefinedLocal, m.UnusedVariable)