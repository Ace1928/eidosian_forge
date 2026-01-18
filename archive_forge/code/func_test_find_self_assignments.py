from os import walk, sep, pardir
from os.path import split, join, abspath, exists, isfile
from glob import glob
import re
import random
import ast
from sympy.testing.pytest import raises
from sympy.testing.quality_unicode import _test_this_file_encoding
def test_find_self_assignments():
    candidates_ok = ['class A(object):\n    def foo(self, arg): arg = self\n', 'class A(object):\n    def foo(self, arg): self.prop = arg\n', 'class A(object):\n    def foo(self, arg): obj, obj2 = arg, self\n', 'class A(object):\n    @classmethod\n    def bar(cls, arg): arg = cls\n', 'class A(object):\n    def foo(var, arg): arg = var\n']
    candidates_fail = ['class A(object):\n    def foo(self, arg): self = arg\n', 'class A(object):\n    def foo(self, arg): obj, self = arg, arg\n', 'class A(object):\n    def foo(self, arg):\n        if arg: self = arg', 'class A(object):\n    @classmethod\n    def foo(cls, arg): cls = arg\n', 'class A(object):\n    def foo(var, arg): var = arg\n']
    for c in candidates_ok:
        assert find_self_assignments(c) == []
    for c in candidates_fail:
        assert find_self_assignments(c) != []