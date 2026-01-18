import os
import tempfile
from fire import __main__
from fire import testutils
def testFileNameFailure(self):
    with self.assertRaises(ValueError):
        __main__.main(['__main__.py', self.file2.name, 'Foo', 'double', '--n', '2'])