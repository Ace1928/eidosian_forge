from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_ignoredInFunction(self):
    """
        An C{__all__} definition does not suppress unused import warnings in a
        function scope.
        """
    self.flakes('\n        def foo():\n            import bar\n            __all__ = ["bar"]\n        ', m.UnusedImport, m.UnusedVariable)