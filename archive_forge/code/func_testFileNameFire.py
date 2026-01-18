import os
import tempfile
from fire import __main__
from fire import testutils
def testFileNameFire(self):
    with self.assertOutputMatches('4'):
        __main__.main(['__main__.py', self.file.name, 'Foo', 'double', '--n', '2'])