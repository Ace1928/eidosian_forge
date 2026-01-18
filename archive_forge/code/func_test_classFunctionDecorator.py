from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_classFunctionDecorator(self):
    """
        Test that shadowing a function definition in a class suite with a
        decorated version of that function does not raise a warning.
        """
    self.flakes('\n        class A:\n            def a(): pass\n            a = classmethod(a)\n        ')