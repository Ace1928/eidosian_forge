from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_functionNamesAreBoundNow(self):
    self.flakes('\n        import fu\n        def fu():\n            fu\n        fu\n        ', m.RedefinedWhileUnused)