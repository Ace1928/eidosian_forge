from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_exceptWithoutNameInFunction(self):
    """
        Don't issue false warning when an unnamed exception is used.
        Previously, there would be a false warning, but only when the
        try..except was in a function
        """
    self.flakes('\n        import tokenize\n        def foo():\n            try: pass\n            except tokenize.TokenError: pass\n        ')