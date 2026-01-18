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
def test_commit_encoding_from_config(self):
    r = self._repo
    c = r.get_config()
    c.set(('i18n',), 'commitEncoding', 'iso8859-1')
    c.write_to_path()
    commit_sha = r.do_commit(b'commit with strange character \xee', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12395, commit_timezone=0, author_timestamp=12395, author_timezone=0)
    self.assertEqual(b'iso8859-1', r[commit_sha].encoding)