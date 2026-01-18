import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_delConditionalNested(self):
    """
        Ignored conditional bindings deletion even if they are nested in other
        blocks.
        """
    self.flakes('\n        context = None\n        test = True\n        if False:\n            with context():\n                del(test)\n        assert(test)\n        ')