import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_delConditional(self):
    """
        Ignores conditional bindings deletion.
        """
    self.flakes('\n        context = None\n        test = True\n        if False:\n            del(test)\n        assert(test)\n        ')