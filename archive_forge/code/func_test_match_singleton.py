from sys import version_info
from pyflakes.test.harness import TestCase, skipIf
def test_match_singleton(self):
    self.flakes("\n            match 1:\n                case True:\n                    print('true')\n        ")