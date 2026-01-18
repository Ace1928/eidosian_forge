from sys import version_info
from pyflakes.test.harness import TestCase, skipIf
def test_match_star(self):
    self.flakes("\n            x = [1, 2, 3]\n            match x:\n                case [1, *y]:\n                    print(f'captured: {y}')\n        ")