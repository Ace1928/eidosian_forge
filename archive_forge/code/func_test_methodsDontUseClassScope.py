from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_methodsDontUseClassScope(self):
    self.flakes('\n        class bar:\n            import fu\n            def fun(self):\n                fu\n        ', m.UndefinedName)