from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf
def test_namedtypes_classes(self):
    self.flakes('\n            from typing import TypedDict, NamedTuple\n            class X(TypedDict):\n                y: TypedDict("z", {"zz":int})\n\n            class Y(NamedTuple):\n                y: NamedTuple("v", [("vv", int)])\n        ')