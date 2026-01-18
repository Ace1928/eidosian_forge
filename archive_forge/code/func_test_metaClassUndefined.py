import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_metaClassUndefined(self):
    self.flakes('\n        from abc import ABCMeta\n        class A(metaclass=ABCMeta): pass\n        ')