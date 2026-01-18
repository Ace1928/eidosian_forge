import os
import tempfile
from fire import __main__
from fire import testutils
def testFileNameModuleFileFailure(self):
    with self.assertRaisesRegex(ValueError, 'Fire can only be called on \\.py files\\.'):
        dirname = os.path.dirname(self.file.name)
        with testutils.ChangeDirectory(dirname):
            with open('foobar', 'w'):
                __main__.main(['__main__.py', 'foobar'])
            os.remove('foobar')