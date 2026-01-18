import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_definedInGenExp(self):
    """
        Using the loop variable of a generator expression results in no
        warnings.
        """
    self.flakes('(a for a in [1, 2, 3] if a)')
    self.flakes('(b for b in (a for a in [1, 2, 3] if a) if b)')