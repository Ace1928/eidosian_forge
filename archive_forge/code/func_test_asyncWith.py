from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_asyncWith(self):
    self.flakes('\n        async def commit(session, data):\n            async with session.transaction():\n                await session.update(data)\n        ')