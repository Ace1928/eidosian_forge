import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_definedAsStarArgs(self):
    """Star and double-star arg names are defined."""
    self.flakes('\n        def f(a, *b, **c):\n            print(a, b, c)\n        ')