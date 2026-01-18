from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_classRedefinedAsFunction(self):
    """
        If a class is redefined as a function, a warning is emitted.
        """
    self.flakes('\n        class Foo:\n            pass\n        def Foo():\n            pass\n        ', m.RedefinedWhileUnused)