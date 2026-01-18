from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_asyncWithItem(self):
    self.flakes('\n        async def commit(session, data):\n            async with session.transaction() as trans:\n                await trans.begin()\n                ...\n                await trans.end()\n        ')