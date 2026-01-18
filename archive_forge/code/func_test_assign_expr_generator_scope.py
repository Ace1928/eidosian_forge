from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_assign_expr_generator_scope(self):
    """Test assignment expressions in generator expressions."""
    self.flakes('\n        if (any((y := x[0]) for x in [[True]])):\n            print(y)\n        ')