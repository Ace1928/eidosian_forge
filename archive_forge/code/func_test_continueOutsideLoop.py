from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_continueOutsideLoop(self):
    self.flakes('\n        continue\n        ', m.ContinueOutsideLoop)
    self.flakes('\n        def f():\n            continue\n        ', m.ContinueOutsideLoop)
    self.flakes('\n        while True:\n            pass\n        else:\n            continue\n        ', m.ContinueOutsideLoop)
    self.flakes('\n        while True:\n            pass\n        else:\n            if 1:\n                if 2:\n                    continue\n        ', m.ContinueOutsideLoop)
    self.flakes('\n        while True:\n            def f():\n                continue\n        ', m.ContinueOutsideLoop)
    self.flakes('\n        while True:\n            class A:\n                continue\n        ', m.ContinueOutsideLoop)