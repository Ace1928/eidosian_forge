import os
from ... import osutils, tests, urlutils
from .. import features, script
def test_import_upstream(self):
    self.run_bzr('init source')
    os.mkdir('source/src')
    with open('source/src/myfile', 'wb') as f:
        f.write(b'hello?')
    os.chdir('source')
    self.run_bzr('add')
    self.run_bzr('commit -m hello')
    self.run_bzr('export ../source-0.1.tar.gz')
    self.run_bzr('export ../source-0.1.tar.bz2')
    self.run_bzr('export ../source-0.1')
    self.run_bzr('init ../import')
    os.chdir('../import')
    self.run_bzr('import ../source-0.1.tar.gz')
    self.assertPathExists('src/myfile')
    result = self.run_bzr('import ../source-0.1.tar.gz', retcode=3)[1]
    self.assertContainsRe(result, 'Working tree has uncommitted changes')
    self.run_bzr('commit -m commit')
    self.run_bzr('import ../source-0.1.tar.gz')
    os.chdir('..')
    self.run_bzr('init import2')
    self.run_bzr('import source-0.1.tar.gz import2')
    self.assertPathExists('import2/src/myfile')
    self.run_bzr('import source-0.1.tar.gz import3')
    self.assertPathExists('import3/src/myfile')
    self.run_bzr('import source-0.1.tar.bz2 import4')
    self.assertPathExists('import4/src/myfile')
    self.run_bzr('import source-0.1 import5')
    self.assertPathExists('import5/src/myfile')