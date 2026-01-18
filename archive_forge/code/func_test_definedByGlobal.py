import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_definedByGlobal(self):
    """
        "global" can make an otherwise undefined name in another function
        defined.
        """
    self.flakes('\n        def a(): global fu; fu = 1\n        def b(): fu\n        ')
    self.flakes('\n        def c(): bar\n        def b(): global bar; bar = 1\n        ')