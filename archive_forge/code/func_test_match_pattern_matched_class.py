from sys import version_info
from pyflakes.test.harness import TestCase, skipIf
def test_match_pattern_matched_class(self):
    self.flakes("\n            from a import B\n\n            match 1:\n                case B(x=1) as y:\n                    print(f'matched {y}')\n        ")
    self.flakes("\n            from a import B\n\n            match 1:\n                case B(a, x=z) as y:\n                    print(f'matched {y} {a} {z}')\n        ")