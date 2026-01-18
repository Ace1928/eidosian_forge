import os
from io import BytesIO
from .. import bedding, ignores
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport
def assertPatternsEquals(self, patterns):
    with open('.bzrignore', 'rb') as f:
        contents = f.read().decode('utf-8').splitlines()
    self.assertEqual(sorted(patterns), sorted(contents))