from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_asyncDefUndefined(self):
    self.flakes('\n        async def bar():\n            return foo()\n        ', m.UndefinedName)