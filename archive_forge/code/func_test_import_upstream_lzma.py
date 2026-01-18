import os
from ... import osutils, tests, urlutils
from .. import features, script
def test_import_upstream_lzma(self):
    self.requireFeature(LzmaFeature)
    self.run_bzr('init source')
    os.mkdir('source/src')
    with open('source/src/myfile', 'wb') as f:
        f.write(b'hello?')
    os.chdir('source')
    self.run_bzr('add')
    self.run_bzr('commit -m hello')
    self.run_bzr('export ../source-0.1.tar.lzma')
    self.run_bzr('export ../source-0.1.tar.xz')
    os.chdir('..')
    self.run_bzr('import source-0.1.tar.lzma import1')
    self.assertPathExists('import1/src/myfile')
    self.run_bzr('import source-0.1.tar.xz import2')
    self.assertPathExists('import2/src/myfile')