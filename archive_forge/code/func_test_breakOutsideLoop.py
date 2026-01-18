from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_breakOutsideLoop(self):
    self.flakes('\n        break\n        ', m.BreakOutsideLoop)
    self.flakes('\n        def f():\n            break\n        ', m.BreakOutsideLoop)
    self.flakes('\n        while True:\n            pass\n        else:\n            break\n        ', m.BreakOutsideLoop)
    self.flakes('\n        while True:\n            pass\n        else:\n            if 1:\n                if 2:\n                    break\n        ', m.BreakOutsideLoop)
    self.flakes('\n        while True:\n            def f():\n                break\n        ', m.BreakOutsideLoop)
    self.flakes('\n        while True:\n            class A:\n                break\n        ', m.BreakOutsideLoop)
    self.flakes('\n        try:\n            pass\n        finally:\n            break\n        ', m.BreakOutsideLoop)