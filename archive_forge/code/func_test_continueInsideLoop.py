from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_continueInsideLoop(self):
    self.flakes('\n        while True:\n            continue\n        ')
    self.flakes('\n        for i in range(10):\n            continue\n        ')
    self.flakes('\n        while True:\n            if 1:\n                continue\n        ')
    self.flakes('\n        for i in range(10):\n            if 1:\n                continue\n        ')
    self.flakes('\n        while True:\n            while True:\n                pass\n            else:\n                continue\n        else:\n            pass\n        ')
    self.flakes('\n        while True:\n            try:\n                pass\n            finally:\n                while True:\n                    continue\n        ')