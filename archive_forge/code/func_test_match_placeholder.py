from sys import version_info
from pyflakes.test.harness import TestCase, skipIf
def test_match_placeholder(self):
    self.flakes("\n            def f():\n                match 1:\n                    case _:\n                        print('catchall!')\n        ")