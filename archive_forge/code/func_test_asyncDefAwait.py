from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_asyncDefAwait(self):
    self.flakes("\n        async def read_data(db):\n            await db.fetch('SELECT ...')\n        ")