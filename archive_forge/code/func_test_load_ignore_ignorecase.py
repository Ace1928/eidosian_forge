import os
import re
import shutil
import tempfile
from io import BytesIO
from dulwich.tests import TestCase
from ..ignore import (
from ..repo import Repo
def test_load_ignore_ignorecase(self):
    tmp_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, tmp_dir)
    repo = Repo.init(tmp_dir)
    config = repo.get_config()
    config.set(b'core', b'ignorecase', True)
    config.write_to_path()
    with open(os.path.join(repo.path, '.gitignore'), 'wb') as f:
        f.write(b'/foo/bar\n')
        f.write(b'/dir\n')
    m = IgnoreFilterManager.from_repo(repo)
    self.assertTrue(m.is_ignored(os.path.join('dir', 'blie')))
    self.assertTrue(m.is_ignored(os.path.join('DIR', 'blie')))