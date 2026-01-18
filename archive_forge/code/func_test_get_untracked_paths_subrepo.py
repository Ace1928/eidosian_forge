import contextlib
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from io import BytesIO, StringIO
from unittest import skipIf
from dulwich import porcelain
from dulwich.tests import TestCase
from ..diff_tree import tree_changes
from ..errors import CommitError
from ..objects import ZERO_SHA, Blob, Tag, Tree
from ..porcelain import CheckoutError
from ..repo import NoIndexPresent, Repo
from ..server import DictBackend
from ..web import make_server, make_wsgi_chain
from .utils import build_commit_graph, make_commit, make_object
def test_get_untracked_paths_subrepo(self):
    with open(os.path.join(self.repo.path, '.gitignore'), 'w') as f:
        f.write('nested/\n')
    with open(os.path.join(self.repo.path, 'notignored'), 'w') as f:
        f.write('blah\n')
    subrepo = Repo.init(os.path.join(self.repo.path, 'nested'), mkdir=True)
    with open(os.path.join(subrepo.path, 'ignored'), 'w') as f:
        f.write('bleep\n')
    with open(os.path.join(subrepo.path, 'with'), 'w') as f:
        f.write('bloop\n')
    with open(os.path.join(subrepo.path, 'manager'), 'w') as f:
        f.write('blop\n')
    self.assertEqual({'.gitignore', 'notignored', os.path.join('nested', '')}, set(porcelain.get_untracked_paths(self.repo.path, self.repo.path, self.repo.open_index())))
    self.assertEqual({'.gitignore', 'notignored'}, set(porcelain.get_untracked_paths(self.repo.path, self.repo.path, self.repo.open_index(), exclude_ignored=True)))
    self.assertEqual({'ignored', 'with', 'manager'}, set(porcelain.get_untracked_paths(subrepo.path, subrepo.path, subrepo.open_index())))
    self.assertEqual(set(), set(porcelain.get_untracked_paths(subrepo.path, self.repo.path, self.repo.open_index())))
    self.assertEqual({os.path.join('nested', 'ignored'), os.path.join('nested', 'with'), os.path.join('nested', 'manager')}, set(porcelain.get_untracked_paths(self.repo.path, subrepo.path, self.repo.open_index())))