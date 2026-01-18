import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_namesDeclaredInExceptBlocks(self):
    """Locals declared in except: blocks can be used after the block.

        This shows the example in test_undefinedExceptionName is
        different."""
    self.flakes("\n        try:\n            raise ValueError('ve')\n        except ValueError as exc:\n            e = exc\n        e\n        ")