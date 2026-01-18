from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_asyncForUnderscoreLoopVar(self):
    self.flakes('\n        async def coro(it):\n            async for _ in it:\n                pass\n        ')