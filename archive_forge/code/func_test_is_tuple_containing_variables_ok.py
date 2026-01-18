from pyflakes.messages import IsLiteral
from pyflakes.test.harness import TestCase
def test_is_tuple_containing_variables_ok(self):
    self.flakes('            x = 5\n            if x is (x,):\n                pass\n        ')