import glob
import locale
import os
import shutil
import stat
import sys
import tempfile
import warnings
from dulwich import errors, objects, porcelain
from dulwich.tests import TestCase, skipIf
from ..config import Config
from ..errors import NotGitRepository
from ..object_store import tree_lookup_path
from ..repo import (
from .utils import open_repo, setup_warning_catcher, tear_down_repo
import sys
from dulwich.repo import Repo
@skipIf(sys.platform in ('win32', 'darwin'), 'tries to implicitly decode as utf8')
def test_commit_no_encode_decode(self):
    r = self._repo
    repo_path_bytes = os.fsencode(r.path)
    encodings = ('utf8', 'latin1')
    names = ['À'.encode(encoding) for encoding in encodings]
    for name, encoding in zip(names, encodings):
        full_path = os.path.join(repo_path_bytes, name)
        with open(full_path, 'wb') as f:
            f.write(encoding.encode('ascii'))
        self.addCleanup(os.remove, full_path)
    r.stage(names)
    commit_sha = r.do_commit(b'Files with different encodings', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12395, commit_timezone=0, author_timestamp=12395, author_timezone=0, ref=None, merge_heads=[self._root_commit])
    for name, encoding in zip(names, encodings):
        mode, id = tree_lookup_path(r.get_object, r[commit_sha].tree, name)
        self.assertEqual(stat.S_IFREG | 420, mode)
        self.assertEqual(encoding.encode('ascii'), r[id].data)