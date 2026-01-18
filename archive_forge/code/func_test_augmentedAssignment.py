from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_augmentedAssignment(self):
    """
        The C{__all__} variable is defined incrementally.
        """
    self.flakes("\n        import a\n        import c\n        __all__ = ['a']\n        __all__ += ['b']\n        if 1 < 3:\n            __all__ += ['c', 'd']\n        ", m.UndefinedExport, m.UndefinedExport)