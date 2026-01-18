from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf
def test_literal_union_type_typing(self):
    self.flakes("\n        from typing import Literal\n\n        def f(x: Literal['some string', 'foo bar']) -> None:\n            return None\n        ")