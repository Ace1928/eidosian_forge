from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_modernProperty(self):
    self.flakes('\n        class A:\n            @property\n            def t(self):\n                pass\n            @t.setter\n            def t(self, value):\n                pass\n            @t.deleter\n            def t(self):\n                pass\n        ')