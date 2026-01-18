from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_assign_expr_nested(self):
    """Test assignment expressions in nested expressions."""
    self.flakes('\n        if ([(y:=x) for x in range(4) if [(z:=q) for q in range(4)]]):\n            print(y)\n            print(z)\n        ')