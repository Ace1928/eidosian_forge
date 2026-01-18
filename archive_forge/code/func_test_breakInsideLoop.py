from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_breakInsideLoop(self):
    self.flakes('\n        while True:\n            break\n        ')
    self.flakes('\n        for i in range(10):\n            break\n        ')
    self.flakes('\n        while True:\n            if 1:\n                break\n        ')
    self.flakes('\n        for i in range(10):\n            if 1:\n                break\n        ')
    self.flakes('\n        while True:\n            while True:\n                pass\n            else:\n                break\n        else:\n            pass\n        ')
    self.flakes('\n        while True:\n            try:\n                pass\n            finally:\n                while True:\n                    break\n        ')
    self.flakes('\n        while True:\n            try:\n                pass\n            finally:\n                break\n        ')
    self.flakes('\n        while True:\n            try:\n                pass\n            finally:\n                if 1:\n                    if 2:\n                        break\n        ')