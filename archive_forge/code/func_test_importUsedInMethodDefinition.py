from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_importUsedInMethodDefinition(self):
    """
        Method named 'foo' with default args referring to module named 'foo'.
        """
    self.flakes('\n        import foo\n\n        class Thing(object):\n            def foo(self, parser=foo.parse_foo):\n                pass\n        ')