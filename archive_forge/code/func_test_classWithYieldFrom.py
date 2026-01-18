from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_classWithYieldFrom(self):
    """
        If a yield from is used inside a class, a warning is emitted.
        """
    self.flakes('\n        class Foo(object):\n            yield from range(10)\n        ', m.YieldOutsideFunction)