from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf
def test_literal_type_some_other_module(self):
    """err on the side of false-negatives for types named Literal"""
    self.flakes("\n        from my_module import compat\n        from my_module.compat import Literal\n\n        def f(x: compat.Literal['some string']) -> None:\n            return None\n        def g(x: Literal['some string']) -> None:\n            return None\n        ")