from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_function_arguments_python3(self):
    self.flakes('\n        def foo(a, b, c=0, *args, d=0, **kwargs):\n            pass\n        ')