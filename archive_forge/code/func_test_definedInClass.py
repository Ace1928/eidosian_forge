import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_definedInClass(self):
    """
        Defined name for generator expressions and dict/set comprehension.
        """
    self.flakes('\n        class A:\n            T = range(10)\n\n            Z = (x for x in T)\n            L = [x for x in T]\n            B = dict((i, str(i)) for i in T)\n        ')
    self.flakes('\n        class A:\n            T = range(10)\n\n            X = {x for x in T}\n            Y = {x:x for x in T}\n        ')