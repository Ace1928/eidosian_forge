from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_classWithReturn(self):
    """
        If a return is used inside a class, a warning is emitted.
        """
    self.flakes('\n        class Foo(object):\n            return\n        ', m.ReturnOutsideFunction)