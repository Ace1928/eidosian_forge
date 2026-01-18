import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_dunderClass(self):
    code = '\n        class Test(object):\n            def __init__(self):\n                print(__class__.__name__)\n                self.x = 1\n\n        t = Test()\n        '
    self.flakes(code)