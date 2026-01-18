import os
import re
import shutil
import tempfile
from io import BytesIO
from dulwich.tests import TestCase
from ..ignore import (
from ..repo import Repo
def test_ignored_contents(self):
    tmp_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, tmp_dir)
    repo = Repo.init(tmp_dir)
    with open(os.path.join(repo.path, '.gitignore'), 'wb') as f:
        f.write(b'a/*\n')
        f.write(b'!a/*.txt\n')
    m = IgnoreFilterManager.from_repo(repo)
    os.mkdir(os.path.join(repo.path, 'a'))
    self.assertIs(None, m.is_ignored('a'))
    self.assertIs(None, m.is_ignored('a/'))
    self.assertFalse(m.is_ignored('a/b.txt'))
    self.assertTrue(m.is_ignored('a/c.dat'))