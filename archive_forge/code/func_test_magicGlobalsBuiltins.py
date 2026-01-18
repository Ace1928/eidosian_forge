import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_magicGlobalsBuiltins(self):
    """
        Use of the C{__builtins__} magic global should not emit an undefined
        name warning.
        """
    self.flakes('__builtins__')