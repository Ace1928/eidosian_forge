from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_functionDecorator(self):
    """
        Test that shadowing a function definition with a decorated version of
        that function does not raise a warning.
        """
    self.flakes('\n        from somewhere import somedecorator\n\n        def a(): pass\n        a = somedecorator(a)\n        ')