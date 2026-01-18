import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_moduleAnnotations(self):
    """
        Use of the C{__annotations__} in module scope should not emit
        an undefined name warning when version is greater than or equal to 3.6.
        """
    self.flakes('__annotations__')