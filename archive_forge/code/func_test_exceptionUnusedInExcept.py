from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_exceptionUnusedInExcept(self):
    self.flakes('\n        try: pass\n        except Exception as e: pass\n        ', m.UnusedVariable)